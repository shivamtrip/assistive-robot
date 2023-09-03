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

#include "whisper/whisper.h"
#include "WhisperParams.hpp"

using namespace alfred;
using namespace interface;

class alfred::interface::WhisperParamsPrivate
{
  /// \brief Whisper full params object
  public: whisper_full_params params;
};

//////////////////////////////////////////////////
WhisperParams::WhisperParams(/* args */)
  : dataPtr(new WhisperParamsPrivate())
{
}

//////////////////////////////////////////////////
WhisperParams::~WhisperParams()
{
}

//////////////////////////////////////////////////
whisper_full_params WhisperParams::Params()
{
  return this->dataPtr->params;
}
