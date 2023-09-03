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

#ifndef ALFRED_SPEECH_RECOGNITION_AUDIO_HPP
#define ALFRED_SPEECH_RECOGNITION_AUDIO_HPP

#include <memory>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_audio.h>

namespace alfred
{
namespace interface
{
// forward declaration
class AudioPrivate;

/// \class Audio Audio.hpp interface/speech_recognition/include/Audio.hpp
class Audio
{
    /// \brief Constructor
    /// \param[in] _lenMs Length of audio data to store in milliseconds
    public: Audio(int _lenMs);
    /// \brief Destructor
    public: ~Audio();

    /// \brief Initialize audio capture
    /// \param[in] _captureId Audio capture device id
    /// \param[in] _sampleRate Sample rate
    /// \return True if successful else false
    public: bool Initalize(int _captureId, int _sampleRate);

    /// \brief Resume audio capture
    /// \return True if successful else false
    public: bool Resume();

    /// \brief Pause audio capture
    /// \return True if successful else false
    public: bool Pause();

    /// \brief Clear audio capture
    /// \return True if successful else false
    public: bool Clear();

    /// \brief Close audio capture
    /// \return True if successful else false
    public: bool Close();

    /// \brief Callback to be called by SDL
    /// \param[in] _stream Audio stream
    /// \param[in] _len Length of audio stream
    void Callback(uint8_t * _stream, int _len);

    /// \brief get audio data from the circular buffer
    /// \param[in] _ms Length of audio data to get in milliseconds
    /// \param[out] _result Audio data
    void Get(int _ms, std::vector<float> & _result);

    /// \brief Private data pointer
    private: std::unique_ptr<AudioPrivate> dataPtr;
};
}  // namespace interface
}  // namespace alfred

#endif  // ALFRED_SPEECH_RECOGNITION_AUDIO_HPP
