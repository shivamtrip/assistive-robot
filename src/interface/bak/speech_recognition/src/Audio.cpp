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
#include <memory>
#include <vector>

#include <ros/ros.h>

#include "Audio.hpp"

using namespace alfred;
using namespace interface;

class alfred::interface::AudioPrivate
{
  // \brief Private data pointer
  public: std::unique_ptr<AudioPrivate> dataPtr;

  // \brief Device id of audio capture device
  public: SDL_AudioDeviceID deviceId = 0;

  // \brief Length of audio data to store in milliseconds
  public: int lenMs = 0;

  // \brief Sample rate of audio capture device
  public: int sampleRate = 0;

  // \brief Flag to indicate if audio capture is running
  public: bool running = false;

  // \brief Mutex to protect audio data
  public: std::mutex mutex;

  // \brief Audio data
  public: std::vector<float> audio;

  // \brief New audio data
  public: std::vector<float> audioNew;

  // \brief Audio data position
  public: size_t audioPosition = 0;

  // \brief Audio data length
  public: size_t audioLen = 0;
};

//////////////////////////////////////////////////
Audio::Audio(int _lenMs)
  : dataPtr(new AudioPrivate())
{
  std::cout << "Audio initialized with duration: " << _lenMs << "\n";
  this->dataPtr->lenMs = _lenMs;
}

//////////////////////////////////////////////////
Audio::~Audio() = default;

//////////////////////////////////////////////////
bool Audio::Initalize(int _captureId, int _sampleRate)
{
  this->dataPtr->sampleRate = _sampleRate;
  SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

  if (SDL_Init(SDL_INIT_AUDIO) < 0)
  {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n", SDL_GetError());
    return false;
  }

  SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);
  {
    int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
    fprintf(stderr, "%s: found %d capture devices:\n", __func__, nDevices);
    for (int i = 0; i < nDevices; i++)
    {
      fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i, SDL_GetAudioDeviceName(i, SDL_TRUE));
    }
  }

  SDL_AudioSpec captureSpecRequested;
  SDL_AudioSpec captureSpecObtained;

  SDL_zero(captureSpecRequested);
  SDL_zero(captureSpecObtained);

  captureSpecRequested.freq = this->dataPtr->sampleRate;
  captureSpecRequested.format = AUDIO_F32;
  captureSpecRequested.channels = 1;
  captureSpecRequested.samples = 1024;
  captureSpecRequested.callback =
  [](void * userdata, uint8_t * stream, int len)
  {
    Audio * audio = reinterpret_cast<Audio *>(userdata);
    audio->Callback(stream, len);
  };
  captureSpecRequested.userdata = this;

  if (_captureId >= 0)
  {
    fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n", __func__,
        _captureId, SDL_GetAudioDeviceName(_captureId, SDL_TRUE));
    this->dataPtr->deviceId = SDL_OpenAudioDevice(
        SDL_GetAudioDeviceName(_captureId, SDL_TRUE), SDL_TRUE,
            &captureSpecRequested, &captureSpecObtained, 0);
  }
  else
  {
    fprintf(stderr, "%s: attempt to open default capture device ...\n", __func__);
    this->dataPtr->deviceId = SDL_OpenAudioDevice(
        nullptr, SDL_TRUE, &captureSpecRequested, &captureSpecObtained, 0);
  }

  if (!this->dataPtr->deviceId)
  {
    fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n",
        __func__, SDL_GetError());
    this->dataPtr->deviceId = 0;

    return false;
  }
  else
  {
    fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n",
        __func__, this->dataPtr->deviceId);
    fprintf(stderr, "%s:     - sample rate:       %d\n",
        __func__, captureSpecObtained.freq);
    fprintf(stderr, "%s:     - format:            %d (required: %d)\n",
        __func__, captureSpecObtained.format,
            captureSpecRequested.format);
    fprintf(stderr, "%s:     - channels:          %d (required: %d)\n",
        __func__, captureSpecObtained.channels,
            captureSpecRequested.channels);
    fprintf(stderr, "%s:     - samples per frame: %d\n",
        __func__, captureSpecObtained.samples);
  }

  this->dataPtr->sampleRate = captureSpecObtained.freq;

  this->dataPtr->audio.resize(
      (this->dataPtr->sampleRate*this->dataPtr->lenMs)/1000);

  return true;
}

//////////////////////////////////////////////////
bool Audio::Resume()
{
  if (!this->dataPtr->deviceId)
  {
    fprintf(stderr, "%s: no audio device to resume!\n", __func__);
    return false;
  }

  if (this->dataPtr->running)
  {
    fprintf(stderr, "%s: already running!\n", __func__);
    return false;
  }

  SDL_PauseAudioDevice(this->dataPtr->deviceId, 0);

  this->dataPtr->running = true;

  return true;
}

//////////////////////////////////////////////////
bool Audio::Pause()
{
  if (!this->dataPtr->deviceId)
  {
    fprintf(stderr, "%s: no audio device to pause!\n", __func__);
    return false;
  }

  if (!this->dataPtr->running)
  {
    fprintf(stderr, "%s: already paused!\n", __func__);
    return false;
  }

  SDL_PauseAudioDevice(this->dataPtr->deviceId, 1);

  this->dataPtr->running = false;

  return true;
}

//////////////////////////////////////////////////
bool Audio::Clear()
{
  if (!this->dataPtr->deviceId)
  {
    fprintf(stderr, "%s: no audio device to clear!\n", __func__);
    return false;
  }

  if (!this->dataPtr->running)
  {
    fprintf(stderr, "%s: not running!\n", __func__);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(this->dataPtr->mutex);

    this->dataPtr->audioPosition = 0;
    this->dataPtr->audioLen = 0;
  }

  return true;
}

//////////////////////////////////////////////////
bool Audio::Close()
{
  if (!this->dataPtr->deviceId)
  {
    fprintf(stderr, "%s: no audio device to close!\n", __func__);
    return false;
  }

  if (this->dataPtr->running)
  {
    fprintf(stderr, "%s: stopping audio device ...\n", __func__);
    SDL_PauseAudioDevice(this->dataPtr->deviceId, 1);
  }

  fprintf(stderr, "%s: closing audio device ...\n", __func__);
  SDL_CloseAudioDevice(this->dataPtr->deviceId);

  this->dataPtr->deviceId = 0;
  this->dataPtr->running = false;

  return true;
}

//////////////////////////////////////////////////
void Audio::Callback(uint8_t *_stream, int _len)
{
  if (!this->dataPtr->running)
  {
        return;
  }

  const size_t nSamples = _len / sizeof(float);

  this->dataPtr->audioNew.resize(nSamples);
  memcpy(this->dataPtr->audioNew.data(), _stream, nSamples * sizeof(float));

  {
    std::lock_guard<std::mutex> lock(this->dataPtr->mutex);

    if (this->dataPtr->audioPosition + nSamples > this->dataPtr->audio.size())
    {
      const size_t n0 = this->dataPtr->audio.size() - this->dataPtr->audioPosition;

      memcpy(&this->dataPtr->audio[this->dataPtr->audioPosition], _stream, n0 * sizeof(float));
      memcpy(&this->dataPtr->audio[0], &_stream[n0], (nSamples - n0) * sizeof(float));

      this->dataPtr->audioPosition = (this->dataPtr->audioPosition + nSamples) % this->dataPtr->audio.size();
      this->dataPtr->audioLen = this->dataPtr->audio.size();
    }
    else
    {
      memcpy(&this->dataPtr->audio[this->dataPtr->audioPosition], _stream, nSamples * sizeof(float));

      this->dataPtr->audioPosition = (this->dataPtr->audioPosition + nSamples) % this->dataPtr->audio.size();
      this->dataPtr->audioLen = std::min(this->dataPtr->audioLen + nSamples, this->dataPtr->audio.size());
    }
  }
}

//////////////////////////////////////////////////
void Audio::Get(int _ms, std::vector<float> &_result)
{
  if (!this->dataPtr->deviceId)
  {
    fprintf(stderr, "%s: no audio device to get audio from!\n", __func__);
    return;
  }

  if (!this->dataPtr->running)
  {
    fprintf(stderr, "%s: not running!\n", __func__);
    return;
  }

  _result.clear();

  {
    std::lock_guard<std::mutex> lock(this->dataPtr->mutex);

    if (_ms <= 0)
    {
      _ms = this->dataPtr->lenMs;
    }

    size_t nSamples = (this->dataPtr->sampleRate * _ms) / 1000;
    if (nSamples > this->dataPtr->audioLen)
    {
      nSamples = this->dataPtr->audioLen;
    }

    _result.resize(nSamples);

    int s0 = this->dataPtr->audioPosition - nSamples;
    if (s0 < 0)
    {
      s0 += this->dataPtr->audio.size();
    }

    if (s0 + nSamples > this->dataPtr->audio.size())
    {
      const size_t n0 = this->dataPtr->audio.size() - s0;

      memcpy(_result.data(), &this->dataPtr->audio[s0], n0 * sizeof(float));
      memcpy(&_result[n0], &this->dataPtr->audio[0], (nSamples - n0) * sizeof(float));
    }
    else
    {
      memcpy(_result.data(), &this->dataPtr->audio[s0], nSamples * sizeof(float));
    }
  }
}
