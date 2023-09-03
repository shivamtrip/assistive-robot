/*
 * Copyright (C) 2022 Auxilio Robotics
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef ALFRED_SPEECH_RECOGNITION_WHISPERWRAPPER_HPP
#define ALFRED_SPEECH_RECOGNITION_WHISPERWRAPPER_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "whisper/whisper.h"

namespace alfred
{
namespace interface
{
// forward declaration
class WhisperWrapperPrivate;
/// \class WhisperWrapper WhisperWrapper.hpp interface/speech_recognition/include/WhisperWrapper.hpp
class WhisperWrapper
{
  /// \brief Constructor
  /// \param[in] _modelPath Path to whisper model
  public: WhisperWrapper(std::string _modelPath);

  /// \brief Destructor
  public: ~WhisperWrapper();

  /// \brief Initialize whisper
  /// \param[in] _params Whisper params object
  /// \param[in] _captureId Capture id
  public: void Initialize(whisper_full_params _params, const int _captureId);

  /// \brief Set commands
  /// \param[in] _commands Vector of commands
  public: void SetCommands(
      const std::vector<std::string> &_commands);

  /// \brief Set whisper model path
  /// \param[in] _modelPath Path to whisper model
  public: void SetModelPath(const std::string &_modelPath);

  /// \brief Transcribe a vector of pcm data
  /// \param[in] _pcmf32 Vector of pcm data
  /// \param[out] _prob Probability of the transcription
  /// \param[out] _timeMS Time in milliseconds
  /// \return Transcription
  public: std::string Transcribe(const std::vector<float> & _pcmf32,
      float &_prob, int64_t &_timeMS);

  /// \brief Get allowed commands
  /// \return Vector of allowed commands
  public: std::vector<std::string> Commands() const;

  /// \brief Check if whisper is running
  /// \return True if running, false otherwise
  public: bool IsRunning() const;

  /// \brief Get transcription
  /// \return Transcription
  public: std::string Transcription() const;

  /// \brief Run whisper
  /// \return Detected speech and type of speech
  public: std::map<std::string, std::string> Run();

  /// \brief Private data pointer
  private: std::unique_ptr<WhisperWrapperPrivate> dataPtr;
};
}  // namespace interface
}  // namespace alfred

#endif  // ALFRED_SPEECH_RECOGNITION_WHISPERWRAPPER_HPP
