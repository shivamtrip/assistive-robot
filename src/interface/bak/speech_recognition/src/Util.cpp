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
#include <cmath>
#include <regex>
#include <string>
#include <vector>

#include "Util.hpp"

namespace alfred
{
namespace interface
{
//////////////////////////////////////////////////
std::string trim(const std::string &_s)
{
  std::regex e("^\\s+|\\s+$");
  return std::regex_replace(_s, e, "");
}

//////////////////////////////////////////////////
void highPassFilter(std::vector<float> &_data, float _cutoff, float _sampleRate)
{
  const float rc = 1.0f / (2.0f * M_PI * _cutoff);
  const float dt = 1.0f / _sampleRate;
  const float alpha = dt / (rc + dt);

  float y = _data[0];

  for (size_t i = 1; i < _data.size(); i++)
  {
    y = alpha * (y + _data[i] - _data[i - 1]);
    _data[i] = y;
  }
}

//////////////////////////////////////////////////
bool process_sdl_events()
{
  SDL_Event event;
  while (SDL_PollEvent(&event))
  {
    switch (event.type)
    {
      case SDL_QUIT:
        return false;
        break;
      default:
        break;
    }
  }

    return true;
}

//////////////////////////////////////////////////
bool detectVoiceActivity(std::vector<float> & _pcmf32, int _sampleRate, int _lastMS,
    float _vadThold, float _freqThold, bool _verbose)
{
  const int nSamples = _pcmf32.size();
  const int nSamplesLast = (_sampleRate * _lastMS) / 1000;

  if (nSamplesLast >= nSamples)
  {
    // not enough samples - assume no speech
    return false;
  }

  if (_freqThold > 0.0f)
  {
    highPassFilter(_pcmf32, _freqThold, _sampleRate);
  }

  float energyAll  = 0.0f;
  float energyLast = 0.0f;

  for (int i = 0; i < nSamples; i++)
  {
    energyAll += fabsf(_pcmf32[i]);

    if (i >= nSamples - nSamplesLast)
    {
      energyLast += fabsf(_pcmf32[i]);
    }
  }

  energyAll  /= nSamples;
  energyLast /= nSamplesLast;

  if (_verbose)
  {
    fprintf(stderr,
        "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n",
            __func__, energyAll, energyLast, _vadThold, _freqThold);
  }

  if (energyLast > _vadThold*energyAll)
  {
    return false;
  }

  return true;
}

//////////////////////////////////////////////////
float similarity(const std::string & _s0, const std::string & _s1)
{
  const size_t len0 = _s0.size() + 1;
  const size_t len1 = _s1.size() + 1;

  std::vector<int> col(len1, 0);
  std::vector<int> prevCol(len1, 0);

  for (size_t i = 0; i < len1; i++)
  {
      prevCol[i] = i;
  }

  for (size_t i = 0; i < len0; i++)
  {
    col[0] = i;
    for (size_t j = 1; j < len1; j++)
    {
      col[j] =
          std::min(std::min(1 + col[j - 1], 1 + prevCol[j]),
              prevCol[j - 1] + (_s0[i - 1] == _s1[j - 1] ? 0 : 1));
    }
    col.swap(prevCol);
  }

  const float dist = prevCol[len1 - 1];

  return 1.0f - (dist / std::max(_s0.size(), _s1.size()));
}
}  // namespace interface
}  // namespace alfred
