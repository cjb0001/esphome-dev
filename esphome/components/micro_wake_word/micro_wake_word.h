#pragma once

#ifdef USE_ESP_IDF

#include "preprocessor_settings.h"
#include "streaming_model.h"

#include "esphome/components/microphone/microphone_source.h"

#include "esphome/core/automation.h"
#include "esphome/core/component.h"
#include "esphome/core/ring_buffer.h"

#include <atomic>
#include <freertos/event_groups.h>

#include <frontend.h>
#include <frontend_util.h>

namespace esphome {
namespace micro_wake_word {

enum State {
  STARTING,
  DETECTING_WAKE_WORD,
  STOPPING,
  STOPPED,
};

class MicroWakeWord : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override;
  void dump_config() override;

  void start();
  void stop();

  bool is_running() const { return this->state_ != State::STOPPED; }

  void set_features_step_size(uint8_t step_size) { this->features_step_size_ = step_size; }

  void set_microphone_source(microphone::MicrophoneSource *microphone_source) {
    this->microphone_source_ = microphone_source;
  }

  void set_stop_after_detection(bool stop_after_detection) { this->stop_after_detection_ = stop_after_detection; }

  /// @brief Sets the ring buffer duration for audio capture
  /// @param duration_ms Duration in milliseconds
  ///
  /// Platform-specific defaults:
  /// - ESP32 (single-core): 360ms
  /// - ESP32-S3/C3 (multi-core): 240ms
  ///
  /// The ring buffer temporarily stores audio data from the microphone before it's processed.
  /// Larger values provide more tolerance for CPU scheduling delays and inference spikes,
  /// but consume more RAM. Smaller values reduce memory usage but may cause audio loss if
  /// the inference task falls behind.
  ///
  /// Recommended values:
  /// - ESP32 (single core): 360-480ms
  /// - ESP32-S3/C3 (multi-core): 240-360ms
  void set_ring_buffer_duration(uint32_t duration_ms) {
    if (duration_ms < 100) {
      ESP_LOGW("micro_wake_word", "Ring buffer duration %ums too small, using 100ms", duration_ms);
      duration_ms = 100;
    }
    if (duration_ms > 2000) {
      ESP_LOGW("micro_wake_word", "Ring buffer duration %ums is unusually large (>2s), verify this is intentional",
               duration_ms);
    }
    this->ring_buffer_duration_ms_ = duration_ms;
  }

  Trigger<std::string> *get_wake_word_detected_trigger() const { return this->wake_word_detected_trigger_; }

  void add_wake_word_model(WakeWordModel *model);

#ifdef USE_MICRO_WAKE_WORD_VAD
  void add_vad_model(const uint8_t *model_start, uint8_t probability_cutoff, size_t sliding_window_size,
                     size_t tensor_arena_size);

  // Intended for the voice assistant component to fetch VAD status
  bool get_vad_state() { return this->vad_state_; }
#endif

  // Intended for the voice assistant component to access which wake words are available
  // Since these are pointers to the WakeWordModel objects, the voice assistant component can enable or disable them
  std::vector<WakeWordModel *> get_wake_words();

 protected:
  microphone::MicrophoneSource *microphone_source_{nullptr};
  Trigger<std::string> *wake_word_detected_trigger_ = new Trigger<std::string>();
  State state_{State::STOPPED};

  std::weak_ptr<RingBuffer> ring_buffer_;
  std::vector<WakeWordModel *> wake_word_models_;

#ifdef USE_MICRO_WAKE_WORD_VAD
  std::unique_ptr<VADModel> vad_model_;
  bool vad_state_{false};
#endif

  bool pending_start_{false};
  bool pending_stop_{false};

  bool stop_after_detection_;

  uint8_t features_step_size_;
#if CONFIG_FREERTOS_UNICORE
  uint32_t ring_buffer_duration_ms_{360};  // Single-core ESP32 needs more buffering for task scheduling
#else
  uint32_t ring_buffer_duration_ms_{240};  // Multi-core ESP32-S3/C3 can use smaller buffer
#endif

  // Audio frontend handles generating spectrogram features
  struct FrontendConfig frontend_config_;
  struct FrontendState frontend_state_;

  // Handles managing the stop/state of the inference task
  EventGroupHandle_t event_group_;

  // Used to send messages about the models' states to the main loop
  QueueHandle_t detection_queue_;

  static void inference_task(void *params);
  TaskHandle_t inference_task_handle_{nullptr};

  /// @brief Suspends the inference task
  void suspend_task_();
  /// @brief Resumes the inference task
  void resume_task_();

  void set_state_(State state);

  /// @brief Generates spectrogram features from an input buffer of audio samples
  /// @param audio_buffer (int16_t *) Buffer containing input audio samples
  /// @param samples_available (size_t) Number of samples avaiable in the input buffer
  /// @param features_buffer (int8_t *) Buffer to store generated features
  /// @return (size_t) Number of samples processed from the input buffer
  size_t generate_features_(int16_t *audio_buffer, size_t samples_available,
                            int8_t features_buffer[PREPROCESSOR_FEATURE_SIZE]);

  /// @brief Processes any new probabilities for each model. If any wake word is detected, it will send a DetectionEvent
  /// to the detection_queue_.
  void process_probabilities_();

  /// @brief Deletes each model's TFLite interpreters and frees tensor arena memory.
  void unload_models_();

  /// @brief Runs an inference with each model using the new spectrogram features
  /// @param audio_features (int8_t *) Buffer containing new spectrogram features
  /// @return True if successful, false if any errors were encountered
  bool update_model_probabilities_(const int8_t audio_features[PREPROCESSOR_FEATURE_SIZE]);

  uint32_t zero_transfer_counter_{0};
  uint32_t last_transfer_log_ms_{0};
  uint32_t last_progress_log_ms_{0};
  uint32_t last_dsp_log_ms_{0};
  uint32_t last_ring_buffer_write_log_ms_{0};
  size_t ring_buffer_capacity_bytes_{0};
  size_t microphone_resume_threshold_bytes_{0};
  bool ring_buffer_low_free_logged_{false};
  bool inference_task_wdt_registered_{false};
  std::atomic<bool> microphone_flow_controlled_{false};

  void register_inference_watchdog_();
  void unregister_inference_watchdog_();
  void feed_inference_watchdog_(const char *context);
  void pause_microphone_for_backpressure_(const char *reason);
  void try_resume_microphone_from_backpressure_(size_t ring_free_bytes);
};

}  // namespace micro_wake_word
}  // namespace esphome

#endif  // USE_ESP_IDF
