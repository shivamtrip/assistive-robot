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

#ifndef ALFRED_SPEECH_RECOGNITION_UTIL_HPP
#define ALFRED_SPEECH_RECOGNITION_UTIL_HPP

#include <regex>
#include <string>
#include <vector>

#include <SDL.h>
#include <SDL_audio.h>

namespace alfred
{
namespace interface
{
/// \brief Remove whitespace from the beginning and end of a string
/// \param[in] _s String to trim
/// \return Trimmed string
std::string trim(const std::string &_s);

/// \brief Run a high pass filter on a vector of data
/// \param[in] _data Vector of data to filter
/// \param[in] _cutoff Cutoff frequency
/// \param[in] _sampleRate Sample rate
void highPassFilter(std::vector<float> &_data, float _cutoff, float _sampleRate);

/// \brief Determine if a vector of data contains voice activity
/// \param[in] _pcmf32 Vector of data to analyze
/// \param[in] _sampleRate Sample rate
/// \param[in] _lastMS Last time voice was detected
/// \param[in] _vadThold VAD threshold
/// \param[in] _freqThold Frequency threshold
/// \param[in] _verbose Print debug info
/// \return True if voice activity is detected else false
bool detectVoiceActivity(std::vector<float> & _pcmf32, int _sampleRate, int _lastMS,
    float _vadThold, float _freqThold, bool _verbose);

/// \brief Calculate the similarity between two strings using the Levenshtein distance
/// \param[in] _s0 First string
/// \param[in] _s1 Second string
/// \return Similarity between the two strings
float similarity(const std::string & _s0, const std::string & _s1);

/// \brief Process SDL events
/// \return True if the application should quit, false otherwise
bool process_sdl_events();
}  // namespace interface
}  // namespace alfred

#endif  // ALFRED_SPEECH_RECOGNITION_UTIL_HPP
