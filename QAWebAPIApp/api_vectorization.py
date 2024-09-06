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
from django.core.cache import cache
from QuestionAnswer.Vectorization import Vectorization as QAVectorization

class Vectorization(APIView):
    def get(self, request, *args, **kwargs):
        data = {
                'message': "success.",
                'status': 'success'
            }
        return Response(data, status=status.HTTP_200_OK)
    # When call the http POST request of this api will enter into this.
    # No parameter is required so far.
    # Return a json obj, include DB_version, e.g. {"DB_version":"2024-08-09_22-38-54-910416"}, and with status code 200.
    # Function:
    #   Read *.txt context files from folder "QuestionAnswer/docs/Context"
    #   Generate the vector DB to folder "QuestionAnswer/docs/chroma/{DB_version}"
    #   Return the DB_version of the generated vector DB.
    def post(self, request, *args, **kwargs):
        vctorization = QAVectorization()
        DB_version = vctorization.run()
        data = {
            'message': 'Vctorization DB has been successfully created.',
            'DB_version': DB_version,
            'status': 'success'
        }
        return Response(data, status=status.HTTP_200_OK)