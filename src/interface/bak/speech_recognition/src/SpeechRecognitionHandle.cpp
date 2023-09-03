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

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>

#include <nlohmann/json.hpp>

#include "alfred_msgs/Speech.h"
#include "alfred_msgs/SpeechTrigger.h"

#include "Audio.hpp"
#include "WhisperParams.hpp"
#include "SpeechRecognitionHandle.hpp"

using namespace alfred;
using namespace interface;

class alfred::interface::SpeechRecognitionHandlePrivate
{
  /// Constructor
  /// \param[in] _nodeHandle ROS node handle
  public: SpeechRecognitionHandlePrivate(const ros::NodeHandle &_nodeHandle);

  /// \brief Trigger word detection callback
  /// \param[in] _msg Trigger word detection message
  public: void TriggerCallback(
    const alfred_msgs::SpeechTrigger::ConstPtr &_msg);

  /// \brief ROS node handle
  public: ros::NodeHandle &nodeHandle;

  /// \brief ROS command publisher
  public: ros::Publisher commandPublisher;

  /// \brief ROS engagement publisher
  public: ros::Publisher engagementPublisher;

  /// \brief ROS trigger subscriber
  public: ros::Subscriber triggerSubscriber;

  /// \brief ROS subscriber topic name
  public: std::string topicName;

  /// \brief ROS subscriber queue size
  public: int queueSize;

  /// \brief Wakeword detected
  public: bool wakewordDetected = false;

  /// \brief Whisper wrapper object
  public: WhisperWrapper whisper{""};

  /// \brief Whisper params object
  public: WhisperParams whisperParams;

  /// \brief Allowed whisper commands
  public: std::vector<std::string> commands;

  /// \brief Speech primitives
  public: std::map<std::string,
      std::vector<std::string>> primitives;

  /// \brief Command to primitive map
  public: std::map<std::string,
      std::string> commandToPrimitive;

  /// \brief Primitives file name
  public: std::string primitivesFileName =
      "primitives.json";

  /// \brief Detected speech and type from whisper
  public: std::map<std::string,
      std::string> detectedSpeech;

  /// \brief ROS topic to publish detected commands
  public: std::string commandTopicName =
      "/interface/speech_recognition/command";

  /// \brief ROS topic to publish detected engagements
  public: std::string engagementTopicName =
      "/interface/speech_recognition/engagement";

  /// \brief ROS topic for wakeword trigger detection
  public: std::string wakewordTopicName =
      "/interface/wakeword_detector/trigger";
};

//////////////////////////////////////////////////
SpeechRecognitionHandlePrivate::SpeechRecognitionHandlePrivate(
  const ros::NodeHandle &_nodeHandle)
  : nodeHandle(const_cast<ros::NodeHandle &>(_nodeHandle)),
    topicName("/speech_recognition"),
    queueSize(10)
{
  if (!this->nodeHandle.getParam(
        "primitives_file_name", this->primitivesFileName))
  {
    ROS_WARN("Failed to get primitives file name, "
      "setting to default value: %s",
      this->primitivesFileName.c_str());
  }
  if (!this->nodeHandle.getParam("command_topic_name",
    this->commandTopicName))
  {
    ROS_WARN("Failed to get speech recognition command topic name, "
      "setting to default value: %s", this->commandTopicName.c_str());
  }
  if (!this->nodeHandle.getParam("wakeword_topic_name",
    this->wakewordTopicName))
  {
    ROS_WARN("Failed to get speech trigger topic name, "
      "setting to default value: %s", this->wakewordTopicName.c_str());
  }

  // Load primitives
  std::string primitivesFilePath =
      ros::package::getPath("speech_recognition") +
      "/commands/" + this->primitivesFileName;

  std::ifstream primitivesFile(primitivesFilePath);
  if (!primitivesFile.is_open())
  {
    ROS_ERROR(
        "Failed to open primitives file: %s", primitivesFilePath.c_str());
    return;
  }

  nlohmann::json primitivesJson;
  primitivesFile >> primitivesJson;

  for (auto &elem : primitivesJson.items())
  {
    this->primitives[elem.key()] =
        primitivesJson[elem.key()].get<std::vector<std::string>>();
  }

  // Load allowed commands
  for (auto it = this->primitives.begin(); it != this->primitives.end(); ++it)
  {
    for (auto &cmd : it->second)
    {
      this->commands.push_back(cmd);
    }
  }

  // Load command to primitive map
  for (auto it = this->primitives.begin(); it != this->primitives.end(); ++it)
  {
    for (auto &cmd : it->second)
    {
      this->commandToPrimitive[cmd] = it->first;
    }
  }

  // Initialize ROS subscriber
  this->triggerSubscriber = this->nodeHandle.subscribe(
      this->wakewordTopicName, this->queueSize,
      &SpeechRecognitionHandlePrivate::TriggerCallback, this);
}

//////////////////////////////////////////////////
void SpeechRecognitionHandlePrivate::TriggerCallback(
    const alfred_msgs::SpeechTrigger::ConstPtr &_msg)
{
  if (_msg->word == "Hey Alfred")
  {
    this->wakewordDetected = true;
  }
}

//////////////////////////////////////////////////
SpeechRecognitionHandle::SpeechRecognitionHandle(
    const ros::NodeHandle &_nodeHandle)
  : dataPtr(new SpeechRecognitionHandlePrivate(_nodeHandle))
{
  // Set whisper model path
  std::string modelPath = ros::package::getPath("speech_recognition") +
      "/models" + "/ggml-base.en.bin";
  this->dataPtr->whisper.SetModelPath(modelPath);

  // Set allowed commands
  this->dataPtr->whisper.SetCommands(this->dataPtr->commands);

  // Initialize whisper
  whisper_full_params params = this->dataPtr->whisperParams.Params();
  this->dataPtr->whisper.Initialize(params, 0);

  // Initialize ROS command publisher
  this->dataPtr->commandPublisher =
      this->dataPtr->nodeHandle.advertise<alfred_msgs::Speech>(
          this->dataPtr->commandTopicName, this->dataPtr->queueSize);

  // Initialize ROS enagement publisher
  this->dataPtr->engagementPublisher =
      this->dataPtr->nodeHandle.advertise<alfred_msgs::Speech>(
          this->dataPtr->engagementTopicName, this->dataPtr->queueSize);
}

//////////////////////////////////////////////////
SpeechRecognitionHandle::~SpeechRecognitionHandle()
{
}

//////////////////////////////////////////////////
void SpeechRecognitionHandle::Run()
{
  // Run whisper only if wakeword is detected
  if (this->dataPtr->wakewordDetected)
  {
    auto start = std::chrono::high_resolution_clock::now();
    this->dataPtr->detectedSpeech = this->dataPtr->whisper.Run();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

  // Publish detected speech to ROS topics
  this->PublishToTopics();
  this->dataPtr->detectedSpeech.clear();
  }

  // Reset wakeword detection
  this->dataPtr->wakewordDetected = false;
}

//////////////////////////////////////////////////
void SpeechRecognitionHandle::PublishToTopics() const
{
  // std::cout << "Publishing: " << this->dataPtr->detectedSpeech["speech"] << "\n";
  alfred_msgs::Speech msg;

  if (this->dataPtr->detectedSpeech["type"] == "command")
  {
    msg.type = "command";
    msg.speech = this->dataPtr->detectedSpeech["speech"];
    msg.primitive = this->dataPtr->commandToPrimitive[msg.speech];

    this->dataPtr->commandPublisher.publish(msg);
  }
  else if (this->dataPtr->detectedSpeech["type"] == "engagement")
  {
    msg.type = "engagement";
    msg.speech = this->dataPtr->detectedSpeech["speech"];
    msg.primitive = "null";

    this->dataPtr->engagementPublisher.publish(msg);
  }
  else
  {
    return;
  }
}
