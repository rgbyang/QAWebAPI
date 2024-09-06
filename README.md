## Getting Started
1. Put the expected context *.txt files to folder "QuestionAnswer\docs\Context"
2. Run this project by Django
3. Access Web API "api/vectorization/" (the default Django dev server is bound to 127.0.0.1:80000, so by default you need to access this Web API by "http://127.0.0.1:8000/api/vectorization/") by POST to vectorize these context files to local DB, and it will return the DB_version. Be default the context vector DB is stored to folder "QuestionAnswer\docs\chroma"
4. Access Web API "api/chat/" by POST to init a chat obj, with json parameter like {"action": "init", "DB_version": "2024-08-09_22-38-54-910416"}. This will return a chat_id in a json obj like {"chat_id": "1b8ee4db-9b8e-4b14-b7f3-fc4165fe2ab8"}.
5. Access Web API "api/chat/" by POST to ask question to a chat obj, with json parameter like {"action": "ask", "chat_id": "1b8ee4db-9b8e-4b14-b7f3-fc4165fe2ab8", "question": ""}. This will try to answer you according to the LLM and the context vector DB. It will return the answer like {"answer": ""}.
6. You can ask question to the same chat id, or init a new chat obj then ask question to the new chat obj. Each chat object will records its own conversation.
7. For details parameters description please refer the description in each Web API code file (api_*.py files under folder QAWebAPIApp).

## License

This project is licensed under the Apache License 2.0. 

### Third-Party Licenses

This project uses the following third-party libraries:

- **Library transformers**: Licensed under the Apache License 2.0
- **Library torch**: Licensed under the BSD License
- **Library langchain**: Licensed under the MIT License
- **Library django**: Licensed under the BSD 3-Clause "New" or "Revised" License
- **Library rest_framework**: Licensed under the Encode OSS Ltd license declaration

For more details, see the [LICENSE](LICENSE) file.