#include "micro_wake_word.h"

#ifdef USE_ESP_IDF

#include "esphome/core/application.h"
#include "esphome/core/hal.h"
#include "esphome/core/helpers.h"
#include "esphome/core/log.h"
#include "esp_task_wdt.h"

#include "esphome/components/audio/audio_transfer_buffer.h"

#ifdef USE_OTA
#include "esphome/components/ota/ota_backend.h"
#endif

#include <algorithm>

namespace esphome {
namespace micro_wake_word {

static const char *const TAG = "micro_wake_word";

// Early warning threshold before buffer exhaustion causes audio loss
static constexpr size_t MIC_RING_BUFFER_LOW_FREE_THRESHOLD = 1024;

// Periodic logging intervals to monitor component health without flooding logs
static constexpr uint32_t INFERENCE_PROGRESS_LOG_INTERVAL_MS = 5000;  // Log every 5 seconds at VERBOSE level
static constexpr uint32_t DSP_PROGRESS_LOG_INTERVAL_MS = 10000;       // Log every 10 seconds at VERBOSE level

// FreeRTOS queue depth for wake word detection events (rare occurrences, small queue sufficient)
static const ssize_t DETECTION_QUEUE_LENGTH = 5;

// Maximum wait time for audio data transfer; matches ~20Hz update rate for responsive audio processing
static const size_t DATA_TIMEOUT_MS = 50;

// FreeRTOS task stack size: 3KB is standard for ESP32 TensorFlow Lite Micro inference tasks
static const uint32_t INFERENCE_TASK_STACK_SIZE = 3072;

// FreeRTOS task priority: 3 is medium priority (higher than idle, lower than critical system tasks)
static const UBaseType_t INFERENCE_TASK_PRIORITY = 3;

enum EventGroupBits : uint32_t {
  COMMAND_STOP = (1 << 0),  // Signals the inference task should stop

  TASK_STARTING = (1 << 3),
  TASK_RUNNING = (1 << 4),
  TASK_STOPPING = (1 << 5),
  TASK_STOPPED = (1 << 6),

  ERROR_MEMORY = (1 << 9),
  ERROR_INFERENCE = (1 << 10),

  WARNING_FULL_RING_BUFFER = (1 << 13),

  ERROR_BITS = ERROR_MEMORY | ERROR_INFERENCE,
  ALL_BITS = 0xfffff,  // 24 total bits available in an event group
};

float MicroWakeWord::get_setup_priority() const { return setup_priority::AFTER_CONNECTION; }

static const LogString *micro_wake_word_state_to_string(State state) {
  switch (state) {
    case State::STARTING:
      return LOG_STR("STARTING");
    case State::DETECTING_WAKE_WORD:
      return LOG_STR("DETECTING_WAKE_WORD");
    case State::STOPPING:
      return LOG_STR("STOPPING");
    case State::STOPPED:
      return LOG_STR("STOPPED");
    default:
      return LOG_STR("UNKNOWN");
  }
}

void MicroWakeWord::dump_config() {
  ESP_LOGCONFIG(TAG, "microWakeWord:");
  ESP_LOGCONFIG(TAG, "  Ring buffer duration: %" PRIu32 " ms", this->ring_buffer_duration_ms_);
  ESP_LOGCONFIG(TAG, "  models:");
  for (auto &model : this->wake_word_models_) {
    model->log_model_config();
  }
#ifdef USE_MICRO_WAKE_WORD_VAD
  this->vad_model_->log_model_config();
#endif
}

void MicroWakeWord::setup() {
  this->frontend_config_.window.size_ms = FEATURE_DURATION_MS;
  this->frontend_config_.window.step_size_ms = this->features_step_size_;
  this->frontend_config_.filterbank.num_channels = PREPROCESSOR_FEATURE_SIZE;
  this->frontend_config_.filterbank.lower_band_limit = FILTERBANK_LOWER_BAND_LIMIT;
  this->frontend_config_.filterbank.upper_band_limit = FILTERBANK_UPPER_BAND_LIMIT;
  this->frontend_config_.noise_reduction.smoothing_bits = NOISE_REDUCTION_SMOOTHING_BITS;
  this->frontend_config_.noise_reduction.even_smoothing = NOISE_REDUCTION_EVEN_SMOOTHING;
  this->frontend_config_.noise_reduction.odd_smoothing = NOISE_REDUCTION_ODD_SMOOTHING;
  this->frontend_config_.noise_reduction.min_signal_remaining = NOISE_REDUCTION_MIN_SIGNAL_REMAINING;
  this->frontend_config_.pcan_gain_control.enable_pcan = PCAN_GAIN_CONTROL_ENABLE_PCAN;
  this->frontend_config_.pcan_gain_control.strength = PCAN_GAIN_CONTROL_STRENGTH;
  this->frontend_config_.pcan_gain_control.offset = PCAN_GAIN_CONTROL_OFFSET;
  this->frontend_config_.pcan_gain_control.gain_bits = PCAN_GAIN_CONTROL_GAIN_BITS;
  this->frontend_config_.log_scale.enable_log = LOG_SCALE_ENABLE_LOG;
  this->frontend_config_.log_scale.scale_shift = LOG_SCALE_SCALE_SHIFT;

  this->event_group_ = xEventGroupCreate();
  if (this->event_group_ == nullptr) {
    ESP_LOGE(TAG, "Failed to create event group");
    this->mark_failed();
    return;
  }

  this->detection_queue_ = xQueueCreate(DETECTION_QUEUE_LENGTH, sizeof(DetectionEvent));
  if (this->detection_queue_ == nullptr) {
    ESP_LOGE(TAG, "Failed to create detection event queue");
    this->mark_failed();
    return;
  }

  this->microphone_source_->add_data_callback([this](const std::vector<uint8_t> &data) {
    if (this->state_ == State::STOPPED) {
      return;
    }
    const auto use_count = this->ring_buffer_.use_count();
    if (use_count == 0) {
      ESP_LOGW(TAG,
               "Microphone produced %u bytes but inference task not ready "
               "(use_count=%zu state=%s)",
               static_cast<unsigned>(data.size()), use_count,
               LOG_STR_ARG(micro_wake_word_state_to_string(this->state_)));
      return;
    }
    std::shared_ptr<RingBuffer> temp_ring_buffer = this->ring_buffer_.lock();
    if (temp_ring_buffer == nullptr) {
      ESP_LOGW(TAG, "Microphone ring buffer expired before write");
      return;
    }
    size_t bytes_free = temp_ring_buffer->free();

    if (bytes_free < data.size()) {
      ESP_LOGW(TAG,
               "Microphone ring buffer overflow (free=%zu incoming=%u), "
               "pausing microphone to relieve backpressure",
               bytes_free, static_cast<unsigned>(data.size()));
      xEventGroupSetBits(this->event_group_, EventGroupBits::WARNING_FULL_RING_BUFFER);
      this->ring_buffer_low_free_logged_ = true;
      if (this->microphone_source_->is_passive()) {
        ESP_LOGW(TAG, "Microphone source is passive; cannot pause capture. "
                      "Dropping audio frame");
        return;
      }
      this->pause_microphone_for_backpressure_("ring buffer overflow");
      return;
    } else if (bytes_free < MIC_RING_BUFFER_LOW_FREE_THRESHOLD) {
      if (!this->ring_buffer_low_free_logged_) {
        ESP_LOGD(TAG, "Microphone ring buffer low free bytes (free=%zu threshold=%u)", bytes_free,
                 static_cast<unsigned>(MIC_RING_BUFFER_LOW_FREE_THRESHOLD));
        this->ring_buffer_low_free_logged_ = true;
      }
    } else if (this->ring_buffer_low_free_logged_) {
      this->ring_buffer_low_free_logged_ = false;
    }
    size_t written = temp_ring_buffer->write((void *) data.data(), data.size());
    if (written != data.size()) {
      ESP_LOGD(TAG, "Partial write: wrote %zu of %u microphone bytes (buffer nearly full)", written,
               static_cast<unsigned>(data.size()));
      // Partial write indicates buffer is nearly full
      // The low free threshold warning and backpressure system will handle this
    }
    const uint32_t now = millis();
    if (now - this->last_ring_buffer_write_log_ms_ >= 5000) {
      ESP_LOGV(TAG, "Mic->mww ring write=%u avail=%zu free=%zu use_count=%zu", static_cast<unsigned>(written),
               temp_ring_buffer->available(), temp_ring_buffer->free(), this->ring_buffer_.use_count());
      this->last_ring_buffer_write_log_ms_ = now;
    }
  });

#ifdef USE_OTA
  ota::get_global_ota_callback()->add_on_state_callback(
      [this](ota::OTAState state, float progress, uint8_t error, ota::OTAComponent *comp) {
        if (state == ota::OTA_STARTED) {
          this->suspend_task_();
        } else if (state == ota::OTA_ERROR) {
          this->resume_task_();
        }
      });
#endif
}

void MicroWakeWord::inference_task(void *params) {
  MicroWakeWord *this_mww = (MicroWakeWord *) params;

  xEventGroupSetBits(this_mww->event_group_, EventGroupBits::TASK_STARTING);

  {  // Ensures any C++ objects fall out of scope to deallocate before deleting the task

    const size_t new_bytes_to_process =
        this_mww->microphone_source_->get_audio_stream_info().ms_to_bytes(this_mww->features_step_size_);
    std::unique_ptr<audio::AudioSourceTransferBuffer> audio_buffer;
    int8_t features_buffer[PREPROCESSOR_FEATURE_SIZE];

    if (!(xEventGroupGetBits(this_mww->event_group_) & ERROR_BITS)) {
      // Allocate audio transfer buffer
      audio_buffer = audio::AudioSourceTransferBuffer::create(new_bytes_to_process);

      if (audio_buffer == nullptr) {
        xEventGroupSetBits(this_mww->event_group_, EventGroupBits::ERROR_MEMORY);
      }
    }

    if (!(xEventGroupGetBits(this_mww->event_group_) & ERROR_BITS)) {
      // Allocate ring buffer
      const size_t ring_buffer_size_bytes =
          this_mww->microphone_source_->get_audio_stream_info().ms_to_bytes(this_mww->ring_buffer_duration_ms_);
      std::shared_ptr<RingBuffer> temp_ring_buffer = RingBuffer::create(ring_buffer_size_bytes);
      if (temp_ring_buffer.use_count() == 0) {
        xEventGroupSetBits(this_mww->event_group_, EventGroupBits::ERROR_MEMORY);
      }
      audio_buffer->set_source(temp_ring_buffer);
      this_mww->ring_buffer_ = temp_ring_buffer;
      this_mww->ring_buffer_capacity_bytes_ = ring_buffer_size_bytes;

      // Set microphone resume threshold: 50% of buffer size provides hysteresis to prevent rapid pause/resume cycles
      this_mww->microphone_resume_threshold_bytes_ = std::max(ring_buffer_size_bytes / 2, static_cast<size_t>(1024));
    }

    if (!(xEventGroupGetBits(this_mww->event_group_) & ERROR_BITS)) {
      this_mww->microphone_source_->start();
      xEventGroupSetBits(this_mww->event_group_, EventGroupBits::TASK_RUNNING);
      this_mww->register_inference_watchdog_();

      bool should_exit_inference_loop = false;
      while (!(xEventGroupGetBits(this_mww->event_group_) & COMMAND_STOP) && !should_exit_inference_loop) {
        this_mww->feed_inference_watchdog_("loop-start");
        size_t bytes_transferred =
            audio_buffer->transfer_data_from_source(pdMS_TO_TICKS(DATA_TIMEOUT_MS), /*pre_shift=*/true);
        while (audio_buffer->free() >= new_bytes_to_process) {
          size_t additional_transfer = audio_buffer->transfer_data_from_source(0, /*pre_shift=*/false);
          if (additional_transfer == 0) {
            break;
          }
          bytes_transferred += additional_transfer;
        }
        const uint32_t now = millis();
        const bool should_log_transfer_progress =
            (now - this_mww->last_progress_log_ms_) >= INFERENCE_PROGRESS_LOG_INTERVAL_MS;

        size_t buffer_available = audio_buffer->available();
        size_t buffer_capacity = audio_buffer->capacity();
        size_t buffer_free = audio_buffer->free();
        size_t ring_available = 0;
        size_t ring_free = 0;
        size_t ring_use_count = this_mww->ring_buffer_.use_count();
        if (auto temp_ring_buffer = this_mww->ring_buffer_.lock()) {
          ring_available = temp_ring_buffer->available();
          ring_free = temp_ring_buffer->free();
        }
        this_mww->try_resume_microphone_from_backpressure_(ring_free);

        if (should_log_transfer_progress) {
          this_mww->feed_inference_watchdog_("transfer-log");
          ESP_LOGV(TAG,
                   "Inference transfer_data_from_source=%zu buffer=%zu/%zu free=%zu "
                   "ring avail=%zu free=%zu use_count=%zu",
                   bytes_transferred, buffer_available, buffer_capacity, buffer_free, ring_available, ring_free,
                   ring_use_count);
          this_mww->last_progress_log_ms_ = now;
        } else {
          this_mww->feed_inference_watchdog_("transfer-loop");
        }

        if (bytes_transferred == 0) {
          this_mww->zero_transfer_counter_++;
          if (this_mww->zero_transfer_counter_ == 1 || now - this_mww->last_transfer_log_ms_ > 1000) {
            ESP_LOGD(TAG,
                     "Inference waiting for data (%u zero reads, buffer=%zu/%zu free=%zu, "
                     "ring avail=%zu free=%zu use_count=%zu state=%s)",
                     this_mww->zero_transfer_counter_, buffer_available, buffer_capacity, buffer_free, ring_available,
                     ring_free, ring_use_count, LOG_STR_ARG(micro_wake_word_state_to_string(this_mww->state_)));
            this_mww->last_transfer_log_ms_ = now;
          }
        } else {
          this_mww->zero_transfer_counter_ = 0;
        }

        if (audio_buffer->available() < new_bytes_to_process) {
          // Insufficient data to generate new spectrogram features, read more next iteration
          delay(1);
          continue;
        }

        do {
          const uint32_t dsp_log_now = millis();
          const bool should_log_dsp_progress =
              (dsp_log_now - this_mww->last_dsp_log_ms_) >= DSP_PROGRESS_LOG_INTERVAL_MS;
          if (should_log_dsp_progress) {
            size_t dsp_ring_available = 0;
            size_t dsp_ring_free = 0;
            if (auto temp_ring_buffer = this_mww->ring_buffer_.lock()) {
              dsp_ring_available = temp_ring_buffer->available();
              dsp_ring_free = temp_ring_buffer->free();
            }
            const size_t samples_available = audio_buffer->available() / sizeof(int16_t);
            this_mww->feed_inference_watchdog_("dsp-log");
            ESP_LOGV(TAG,
                     "Inference feeding DSP samples=%zu buffer_bytes=%zu target_bytes=%zu "
                     "ring avail=%zu free=%zu",
                     samples_available, audio_buffer->available(), new_bytes_to_process, dsp_ring_available,
                     dsp_ring_free);
            this_mww->last_dsp_log_ms_ = dsp_log_now;
          } else {
            this_mww->feed_inference_watchdog_("dsp-loop");
          }

          // Generate new spectrogram features
          uint32_t processed_samples =
              this_mww->generate_features_((int16_t *) audio_buffer->get_buffer_start(),
                                           audio_buffer->available() / sizeof(int16_t), features_buffer);
          audio_buffer->decrease_buffer_length(processed_samples * sizeof(int16_t));

          // Run inference using the new spectorgram features
          this_mww->feed_inference_watchdog_("before-inference");
          if (!this_mww->update_model_probabilities_(features_buffer)) {
            xEventGroupSetBits(this_mww->event_group_, EventGroupBits::ERROR_INFERENCE);
            should_exit_inference_loop = true;
            break;
          }
          this_mww->feed_inference_watchdog_("after-inference");

          // Process each model's probabilities and possibly send a Detection Event to the queue
          this_mww->process_probabilities_();
        } while (audio_buffer->available() >= new_bytes_to_process);

        // Yield to other tasks to prevent task starvation; 1ms is the minimum FreeRTOS tick period
        delay(1);
      }
      this_mww->unregister_inference_watchdog_();
    }
  }

  xEventGroupSetBits(this_mww->event_group_, EventGroupBits::TASK_STOPPING);

  this_mww->unload_models_();
  this_mww->microphone_source_->stop();
  FrontendFreeStateContents(&this_mww->frontend_state_);

  xEventGroupSetBits(this_mww->event_group_, EventGroupBits::TASK_STOPPED);
  while (true) {
    // Continuously delay until the main loop deletes the task
    delay(10);
  }
}

std::vector<WakeWordModel *> MicroWakeWord::get_wake_words() {
  std::vector<WakeWordModel *> external_wake_word_models;
  for (auto *model : this->wake_word_models_) {
    if (!model->get_internal_only()) {
      external_wake_word_models.push_back(model);
    }
  }
  return external_wake_word_models;
}

void MicroWakeWord::add_wake_word_model(WakeWordModel *model) { this->wake_word_models_.push_back(model); }

#ifdef USE_MICRO_WAKE_WORD_VAD
void MicroWakeWord::add_vad_model(const uint8_t *model_start, uint8_t probability_cutoff, size_t sliding_window_size,
                                  size_t tensor_arena_size) {
  this->vad_model_ = make_unique<VADModel>(model_start, probability_cutoff, sliding_window_size, tensor_arena_size);
}
#endif

void MicroWakeWord::suspend_task_() {
  if (this->inference_task_handle_ != nullptr) {
    vTaskSuspend(this->inference_task_handle_);
  }
}

void MicroWakeWord::resume_task_() {
  if (this->inference_task_handle_ != nullptr) {
    vTaskResume(this->inference_task_handle_);
  }
}

void MicroWakeWord::register_inference_watchdog_() {
  const esp_err_t wdt_add_result = esp_task_wdt_add(nullptr);
  if (wdt_add_result == ESP_OK) {
    this->inference_task_wdt_registered_ = true;
    ESP_LOGD(TAG, "Registered mww task with watchdog");
  } else {
    ESP_LOGW(TAG, "Failed to add inference task to watchdog err=%d", wdt_add_result);
    this->inference_task_wdt_registered_ = false;
  }
}

void MicroWakeWord::unregister_inference_watchdog_() {
  if (!this->inference_task_wdt_registered_) {
    return;
  }
  const esp_err_t wdt_delete_result = esp_task_wdt_delete(nullptr);
  if (wdt_delete_result != ESP_OK) {
    ESP_LOGW(TAG, "Failed to remove inference task from watchdog err=%d", wdt_delete_result);
  } else {
    ESP_LOGD(TAG, "Unregistered mww task from watchdog");
  }
  this->inference_task_wdt_registered_ = false;
}

void MicroWakeWord::feed_inference_watchdog_(const char *context) {
  if (!this->inference_task_wdt_registered_) {
    return;
  }
  const esp_err_t watchdog_result = esp_task_wdt_reset();
  if (watchdog_result != ESP_OK) {
    ESP_LOGW(TAG, "esp_task_wdt_reset failed err=%d context=%s", watchdog_result, context);
  }
}

void MicroWakeWord::pause_microphone_for_backpressure_(const char *reason) {
  if (this->microphone_flow_controlled_.exchange(true)) {
    return;
  }
  ESP_LOGW(TAG, "Pausing microphone capture: %s", reason);
  this->microphone_source_->stop();
}

void MicroWakeWord::try_resume_microphone_from_backpressure_(size_t ring_free_bytes) {
  if (!this->microphone_flow_controlled_.load()) {
    return;
  }
  if (ring_free_bytes < this->microphone_resume_threshold_bytes_) {
    return;
  }
  ESP_LOGD(TAG, "Resuming microphone capture (ring buffer free=%zu threshold=%zu)", ring_free_bytes,
           this->microphone_resume_threshold_bytes_);
  this->microphone_flow_controlled_ = false;
  this->microphone_source_->start();
}

void MicroWakeWord::loop() {
  uint32_t event_group_bits = xEventGroupGetBits(this->event_group_);

  if (event_group_bits & EventGroupBits::ERROR_MEMORY) {
    xEventGroupClearBits(this->event_group_, EventGroupBits::ERROR_MEMORY);
    ESP_LOGE(TAG, "Encountered an error allocating buffers");
  }

  if (event_group_bits & EventGroupBits::ERROR_INFERENCE) {
    xEventGroupClearBits(this->event_group_, EventGroupBits::ERROR_INFERENCE);
    ESP_LOGE(TAG, "Encountered an error while performing an inference");
  }

  if (event_group_bits & EventGroupBits::WARNING_FULL_RING_BUFFER) {
    xEventGroupClearBits(this->event_group_, EventGroupBits::WARNING_FULL_RING_BUFFER);
    ESP_LOGW(TAG, "Not enough free bytes in ring buffer to store incoming audio data. Resetting the ring buffer. Wake "
                  "word detection accuracy will temporarily be reduced.");
  }

  if (event_group_bits & EventGroupBits::TASK_STARTING) {
    ESP_LOGD(TAG, "Inference task has started, attempting to allocate memory for buffers");
    xEventGroupClearBits(this->event_group_, EventGroupBits::TASK_STARTING);
  }

  if (event_group_bits & EventGroupBits::TASK_RUNNING) {
    ESP_LOGD(TAG, "Inference task is running");

    xEventGroupClearBits(this->event_group_, EventGroupBits::TASK_RUNNING);
    this->set_state_(State::DETECTING_WAKE_WORD);
  }

  if (event_group_bits & EventGroupBits::TASK_STOPPING) {
    ESP_LOGD(TAG, "Inference task is stopping, deallocating buffers");
    xEventGroupClearBits(this->event_group_, EventGroupBits::TASK_STOPPING);
  }

  if ((event_group_bits & EventGroupBits::TASK_STOPPED)) {
    ESP_LOGD(TAG, "Inference task is finished, freeing task resources");
    vTaskDelete(this->inference_task_handle_);
    this->inference_task_handle_ = nullptr;
    xEventGroupClearBits(this->event_group_, ALL_BITS);
    xQueueReset(this->detection_queue_);
    this->set_state_(State::STOPPED);
  }

  if ((this->pending_start_) && (this->state_ == State::STOPPED)) {
    this->set_state_(State::STARTING);
    this->pending_start_ = false;
  }

  if ((this->pending_stop_) && (this->state_ == State::DETECTING_WAKE_WORD)) {
    this->set_state_(State::STOPPING);
    this->pending_stop_ = false;
  }

  switch (this->state_) {
    case State::STARTING:
      if ((this->inference_task_handle_ == nullptr) && !this->status_has_error()) {
        // Setup preprocesor feature generator. If done in the task, it would lock the task to its initial core, as it
        // uses floating point operations.
        if (!FrontendPopulateState(&this->frontend_config_, &this->frontend_state_,
                                   this->microphone_source_->get_audio_stream_info().get_sample_rate())) {
          this->status_momentary_error(
              "Failed to allocate buffers for spectrogram feature processor, attempting again in 1 second", 1000);
          return;
        }

        xTaskCreate(MicroWakeWord::inference_task, "mww", INFERENCE_TASK_STACK_SIZE, (void *) this,
                    INFERENCE_TASK_PRIORITY, &this->inference_task_handle_);

        if (this->inference_task_handle_ == nullptr) {
          FrontendFreeStateContents(&this->frontend_state_);  // Deallocate frontend state
          this->status_momentary_error("Task failed to start, attempting again in 1 second", 1000);
        }
      }
      break;
    case State::DETECTING_WAKE_WORD: {
      DetectionEvent detection_event;
      while (xQueueReceive(this->detection_queue_, &detection_event, 0)) {
        if (detection_event.blocked_by_vad) {
          ESP_LOGD(TAG, "Wake word model predicts '%s', but VAD model doesn't.", detection_event.wake_word->c_str());
        } else {
          constexpr float uint8_to_float_divisor =
              255.0f;  // Converting a quantized uint8 probability to floating point
          ESP_LOGD(TAG, "Detected '%s' with sliding average probability is %.2f and max probability is %.2f",
                   detection_event.wake_word->c_str(), (detection_event.average_probability / uint8_to_float_divisor),
                   (detection_event.max_probability / uint8_to_float_divisor));
          this->wake_word_detected_trigger_->trigger(*detection_event.wake_word);
          if (this->stop_after_detection_) {
            this->stop();
          }
        }
      }
      break;
    }
    case State::STOPPING:
      xEventGroupSetBits(this->event_group_, EventGroupBits::COMMAND_STOP);
      break;
    case State::STOPPED:
      break;
  }
}

void MicroWakeWord::start() {
  if (!this->is_ready()) {
    ESP_LOGW(TAG, "Wake word detection can't start as the component hasn't been setup yet");
    return;
  }

  if (this->is_failed()) {
    ESP_LOGW(TAG, "Wake word component is marked as failed. Please check setup logs");
    return;
  }

  if (this->is_running()) {
    ESP_LOGW(TAG, "Wake word detection is already running");
    return;
  }

  ESP_LOGD(TAG, "Starting wake word detection");

  this->pending_start_ = true;
  this->pending_stop_ = false;
}

void MicroWakeWord::stop() {
  if (this->state_ == STOPPED)
    return;

  ESP_LOGD(TAG, "Stopping wake word detection");

  this->pending_start_ = false;
  this->pending_stop_ = true;
}

void MicroWakeWord::set_state_(State state) {
  if (this->state_ != state) {
    ESP_LOGD(TAG, "State changed from %s to %s", LOG_STR_ARG(micro_wake_word_state_to_string(this->state_)),
             LOG_STR_ARG(micro_wake_word_state_to_string(state)));
    this->state_ = state;
  }
}

size_t MicroWakeWord::generate_features_(int16_t *audio_buffer, size_t samples_available,
                                         int8_t features_buffer[PREPROCESSOR_FEATURE_SIZE]) {
  size_t processed_samples = 0;
  struct FrontendOutput frontend_output =
      FrontendProcessSamples(&this->frontend_state_, audio_buffer, samples_available, &processed_samples);

  for (size_t i = 0; i < frontend_output.size; ++i) {
    // These scaling values are set to match the TFLite audio frontend int8 output.
    // The feature pipeline outputs 16-bit signed integers in roughly a 0 to 670
    // range. In training, these are then arbitrarily divided by 25.6 to get
    // float values in the rough range of 0.0 to 26.0. This scaling is performed
    // for historical reasons, to match up with the output of other feature
    // generators.
    // The process is then further complicated when we quantize the model. This
    // means we have to scale the 0.0 to 26.0 real values to the -128 (INT8_MIN)
    // to 127 (INT8_MAX) signed integer numbers.
    // All this means that to get matching values from our integer feature
    // output into the tensor input, we have to perform:
    // input = (((feature / 25.6) / 26.0) * 256) - 128
    // To simplify this and perform it in 32-bit integer math, we rearrange to:
    // input = (feature * 256) / (25.6 * 26.0) - 128
    constexpr int32_t value_scale = 256;
    constexpr int32_t value_div = 666;  // 666 = 25.6 * 26.0 after rounding
    int32_t value = ((frontend_output.values[i] * value_scale) + (value_div / 2)) / value_div;

    value += INT8_MIN;  // Adds a -128; i.e., subtracts 128
    features_buffer[i] = static_cast<int8_t>(clamp<int32_t>(value, INT8_MIN, INT8_MAX));
  }

  return processed_samples;
}

void MicroWakeWord::process_probabilities_() {
#ifdef USE_MICRO_WAKE_WORD_VAD
  DetectionEvent vad_state = this->vad_model_->determine_detected();

  this->vad_state_ = vad_state.detected;  // atomic write, so thread safe
#endif

  for (auto &model : this->wake_word_models_) {
    if (model->get_unprocessed_probability_status()) {
      // Only detect wake words if there is a new probability since the last check
      DetectionEvent wake_word_state = model->determine_detected();
      if (wake_word_state.detected) {
#ifdef USE_MICRO_WAKE_WORD_VAD
        if (vad_state.detected) {
#endif
          xQueueSend(this->detection_queue_, &wake_word_state, portMAX_DELAY);

          // Wake main loop immediately to process wake word detection
#if defined(USE_SOCKET_SELECT_SUPPORT) && defined(USE_WAKE_LOOP_THREADSAFE)
          App.wake_loop_threadsafe();
#endif

          model->reset_probabilities();
#ifdef USE_MICRO_WAKE_WORD_VAD
        } else {
          wake_word_state.blocked_by_vad = true;
          xQueueSend(this->detection_queue_, &wake_word_state, portMAX_DELAY);
        }
#endif
      }
    }
  }
}

void MicroWakeWord::unload_models_() {
  for (auto &model : this->wake_word_models_) {
    model->unload_model();
  }
#ifdef USE_MICRO_WAKE_WORD_VAD
  this->vad_model_->unload_model();
#endif
}

bool MicroWakeWord::update_model_probabilities_(const int8_t audio_features[PREPROCESSOR_FEATURE_SIZE]) {
  bool success = true;

  for (auto &model : this->wake_word_models_) {
    // Perform inference
    success = success & model->perform_streaming_inference(audio_features);
  }
#ifdef USE_MICRO_WAKE_WORD_VAD
  success = success & this->vad_model_->perform_streaming_inference(audio_features);
#endif

  return success;
}

}  // namespace micro_wake_word
}  // namespace esphome

#endif  // USE_ESP_IDF
