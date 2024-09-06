# Copyright 2024-, RGBYang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from QuestionAnswer.Chat import Chat as QAChat
import uuid
from .global_initializer import global_lock
from .global_initializer import global_chats

class Chat(APIView):
    def get(self, request, *args, **kwargs):
        data = {
                'message': "success.",
                'status': 'success'
            }
        return Response(data, status=status.HTTP_200_OK)
    # When call the http POST request of this api will enter into this.
    # The input parameter is a json obj in the body of the http POST request.
    # Parameter "action" is required
    # When parameter "action" value is "init", e.g. {'action': "init"}
    #   Parameter "model_name" is optional, default value: "bert-base-chinese"
    #   Parameter "basic_DB_path" is optional, default value: 'QuestionAnswer/docs/chroma'
    #   Parameter "DB_version" is optional, default value: "2024-08-09_22-38-54-910416"
    #   Return "chat_id" in json obj, e.g. {'chat_id': "1b8ee4db-9b8e-4b14-b7f3-fc4165fe2ab8"}, and with status code 200.
    #   Function: according to the parameters, init a chat obj, and store it to the global dictionary in the system memory, then return the chat_id.
    # When parameter "action" value is "ask"
    #   Parameter "chat_id" and "question" are required, e.g. {'action': "ask", "chat_id": "1b8ee4db-9b8e-4b14-b7f3-fc4165fe2ab8", "question":"xxx"}
    #   Return "answer" in json obj, e.g. {"answer":"xxx"}, and with status code 200.
    #   Function: will ask quesiton to the specified chat obj, then return the answer.
    def post(self, request, *args, **kwargs):
        input = request.data
        action = input.get('action')
        if action == "init":
            model_name = "bert-base-chinese"
            if input.get('model_name') is not None:
                model_name = input.get('model_name')
            basic_DB_path = 'QuestionAnswer/docs/chroma'
            if input.get('basic_DB_path') is not None:
                basic_DB_path = input.get('basic_DB_path')            
            DB_version = "2024-08-09_22-38-54-910416"#"2024-08-09_23-14-54-775873"
            if input.get('DB_version') is not None:
                DB_version = input.get('DB_version')
            chat_obj = QAChat(model_name, basic_DB_path, DB_version)
            chat_id = uuid.uuid4()
            chat_id_str = str(chat_id)
            with global_lock:
                global_chats[chat_id_str] = chat_obj
            data = {
                'message': 'Chat object has been successfully initialized.',
                'chat_id': chat_id_str,
                'status': 'success'
            }
            return Response(data, status=status.HTTP_200_OK)
        elif action == "ask":
            chat_id = input.get('chat_id')
            question = input.get('question')
            if chat_id is None:
                data = {
                    'message': "'chat_id' can't be empty when action is 'ask'.",
                    'status': 'fail'
                }
                return Response(data, status=status.HTTP_400_BAD_REQUEST)
            if question is None:
                data = {
                    'message': "'question' can't be empty when action is 'ask'.",
                    'status': 'fail'
                }
                return Response(data, status=status.HTTP_400_BAD_REQUEST)
            with global_lock:
                if chat_id in global_chats:
                    chat_obj = global_chats[chat_id]
                else:
                    data = {
                        'message': "Can't find chat_obj from specified 'chat_id'.",
                        'status': 'fail'
                    }
                    return Response(data, status=status.HTTP_404_NOT_FOUND)                
            answer = chat_obj.ask(question)
            data = {
                    'message': "'ask' success.",
                    'answer': answer,
                    'status': 'success'
                }
            return Response(data, status=status.HTTP_200_OK)
        else:
            if action is None:
                action = ''
            data = {
                'message': "'action' paramter value '{}' is not supported.".format(action),
                'status': 'fail'
            }
            return Response(data, status=status.HTTP_400_BAD_REQUEST)