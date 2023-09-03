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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <thread>

#include "whisper/whisper.h"
#include "Audio.hpp"
#include "WhisperWrapper.hpp"
#include "Util.hpp"

using namespace alfred;
using namespace interface;

class alfred::interface::WhisperWrapperPrivate
{
  /// \brief Whisper full params object
  public: whisper_full_params params;

  /// \brief Whisper ctx object
  public: whisper_context * ctx  = nullptr;

  /// \brief Is running
  public: bool isRunning = false;

  /// \brief Have speech
  public: bool haveSpeech = false;

  /// \brief  Allowed commands
  public: std::vector<std::string> commands;

  /// \brief Validation threshold
  public: float validationThreshold = 0.6f;

  /// \brief Frequency threshold
  public: float frequencyThreshold = 100.0f;

  /// \brief Print energy flag
  public: bool printEnergy = false;

  /// \brief Prompt duration in ms
  public: int promptDuration = 5000;

  /// \brief Command duration in ms
  public: int commandDuration = 8000;

  /// \brief Detected speech and type (command or engagement)
  public: std::map<std::string, std::string> detectedSpeech;

  /// \brief Model path
  public: std::string model;

  /// \brief Audio object
  public: std::unique_ptr<Audio> audio;
};

//////////////////////////////////////////////////
WhisperWrapper::WhisperWrapper(std::string _modelPath)
  : dataPtr(new WhisperWrapperPrivate())
{
  this->dataPtr->model = _modelPath;
}

//////////////////////////////////////////////////
WhisperWrapper::~WhisperWrapper()
{
}

//////////////////////////////////////////////////
void WhisperWrapper::Initialize(whisper_full_params _params, int _captureId)
{
  this->dataPtr->params = _params;

  // TODO(atharva-18): Check if model exists
  this->dataPtr->ctx = whisper_init_from_file(this->dataPtr->model.c_str());
  if (!this->dataPtr->ctx)
  {
    fprintf(stderr, "%s: whisper_init() failed!\n", __func__);
    exit(1);
  }
  // TODO(atharva-18): Add translation
  this->dataPtr->audio = std::make_unique<Audio>(30*1000);
  if (!this->dataPtr->audio->Initalize(_captureId, WHISPER_SAMPLE_RATE))
  {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return;
  }

  // TODO(atharva-18): Add error handling
  this->dataPtr->audio->Resume();

  // Wait for 1 second to allow audio to start
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  this->dataPtr->audio->Clear();
}

//////////////////////////////////////////////////
void WhisperWrapper::SetCommands(
    const std::vector<std::string> &_commands)
{
  this->dataPtr->commands = _commands;
}

//////////////////////////////////////////////////
void WhisperWrapper::SetModelPath(const std::string &_modelPath)
{
  this->dataPtr->model = _modelPath;
}

//////////////////////////////////////////////////
std::vector<std::string> WhisperWrapper::Commands() const
{
  return this->dataPtr->commands;
}

//////////////////////////////////////////////////
std::string WhisperWrapper::Transcribe(const std::vector<float> & _pcmf32,
    float &_prob, int64_t &_timeMS)
{
  const auto tStart = std::chrono::high_resolution_clock::now();

  _prob = 0.0f;
  _timeMS = 0;

  whisper_full_params wParams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

  wParams.print_progress   = false;
  wParams.print_special    = false;
  wParams.print_realtime   = false;
  wParams.print_timestamps = false;
  wParams.translate        = false;
  wParams.no_context       = true;
  wParams.single_segment   = true;
  wParams.max_tokens       = 32;
  wParams.language         = std::string{"en"}.c_str();
  wParams.n_threads        = std::min(8, (int32_t) std::thread::hardware_concurrency());

  wParams.audio_ctx        = 0;
  wParams.speed_up         = false;

  if (whisper_full(this->dataPtr->ctx, wParams, _pcmf32.data(), _pcmf32.size()) != 0)
  {
    return "";
  }

  int probN = 0;
  std::string result;

  const int nSegments = whisper_full_n_segments(this->dataPtr->ctx);
  for (int i = 0; i < nSegments; ++i)
  {
    const char * text = whisper_full_get_segment_text(this->dataPtr->ctx, i);
    result += text;

    const int nTokens = whisper_full_n_tokens(this->dataPtr->ctx, i);
    for (int j = 0; j < nTokens; ++j)
    {
      const auto token = whisper_full_get_token_data(this->dataPtr->ctx, i, j);

      _prob += token.p;
      ++probN;
    }
  }

  if (probN > 0)
  {
    _prob /= probN;
  }

  const auto tEnd = std::chrono::high_resolution_clock::now();
  _timeMS = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();

  return result;
}

//////////////////////////////////////////////////
std::map<std::string, std::string> WhisperWrapper::Run()
{
  if (!this->dataPtr->ctx)
  {
    this->dataPtr->detectedSpeech.clear();
    return this->dataPtr->detectedSpeech;
  }

  if (!this->dataPtr->audio)
  {
    this->dataPtr->detectedSpeech.clear();
    return this->dataPtr->detectedSpeech;
  }

  if (this->dataPtr->commands.empty())
  {
    this->dataPtr->detectedSpeech.clear();
    return this->dataPtr->detectedSpeech;
  }

  this->dataPtr->isRunning = true;

  // Wait for 1 second to allow audio to start
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  this->dataPtr->audio->Clear();

  // Run algorithm
  std::string command;
  float prob0 = 0.0f;
  float prob = 0.0f;
  std::vector<float> pcmf32Cur;
  std::vector<float> pcmf32Prompt;
  std::string speech;

  std::cout << "Listening for commands...\n";

  while (!this->dataPtr->haveSpeech)
  {
    this->dataPtr->isRunning = process_sdl_events();

    // delay this thread for 100 ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Get audio data
    this->dataPtr->audio->Get(2000, pcmf32Cur);

    if (detectVoiceActivity(pcmf32Cur, WHISPER_SAMPLE_RATE,
        1000, this->dataPtr->validationThreshold,
        this->dataPtr->frequencyThreshold, this->dataPtr->printEnergy))
    {
      std::cout << "Speech detected detected! processing...\n";
      int64_t tMS = 0;
      {
        // Hear the user command
        // TODO(atharva-18): Might need to tune the duration
        this->dataPtr->audio->Get(this->dataPtr->commandDuration, pcmf32Cur);

        const auto txt = trim(this->Transcribe(pcmf32Cur, prob, tMS));
        std::cout << "Heard: " << txt << "\n";
        prob = 100.0f * (prob - prob0);

        // Match recognized text to the list of commands
        for (const auto & cmd : this->dataPtr->commands)
        {
          const float sim = similarity(txt, cmd);
          if (sim > 0.8f)
          {
            command = cmd;
            break;
          }
        }

        if (command != "")
        {
          fprintf(stdout, "%s: Command '%s%s%s', (t = %d ms)\n",
              __func__, "\033[1m", command.c_str(), "\033[0m", static_cast<int>(tMS));
          fprintf(stdout, "\n");
        }

        speech = txt;
        this->dataPtr->haveSpeech = true;
      }

      this->dataPtr->audio->Clear();
    }
  }

  if (command != "")
  {
    std::cout << "Setting as command type\n";
    this->dataPtr->detectedSpeech["type"] = "command";
    this->dataPtr->detectedSpeech["speech"] = command;
  }
  else
  {
    std::cout << "Setting as engagement type\n";
    this->dataPtr->detectedSpeech["type"] = "engagement";
    this->dataPtr->detectedSpeech["speech"] = speech;
  }

  this->dataPtr->isRunning = false;
  this->dataPtr->haveSpeech = false;

  return this->dataPtr->detectedSpeech;
}
