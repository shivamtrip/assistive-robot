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

#ifndef ALFRED_SPEECH_RECOGNITION_SPEECHRECOGNITIONHANDLE_HPP
#define ALFRED_SPEECH_RECOGNITION_SPEECHRECOGNITIONHANDLE_HPP

#include <memory>

#include <ros/ros.h>

#include "WhisperWrapper.hpp"

namespace alfred
{
namespace interface
{
// forward declaration
class SpeechRecognitionHandlePrivate;

/// \class SpeechRecognitionHandle SpeechRecognitionHandle.hpp
/// interface/speech_recognition/include/SpeechRecognitionHandle.hpp
class SpeechRecognitionHandle
{
  /// \brief Constructor
  public: SpeechRecognitionHandle(const ros::NodeHandle &_nodeHandle);

  /// \brief Destructor
  public: virtual ~SpeechRecognitionHandle();

  /// \brief Run speech recognition
  public: void Run();

  /// \brief Publish data to topics
  public: void PublishToTopics() const;

  /// \brief Private data pointer
  private: std::unique_ptr<SpeechRecognitionHandlePrivate> dataPtr;
};
}  // namespace interface
}  // namespace alfred

#endif  // ALFRED_SPEECH_RECOGNITION_SPEECHRECOGNITIONHANDLE_HPP
