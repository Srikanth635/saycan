import json

from flask import Flask, render_template, request
import os
from openai import OpenAI
from dotenv import load_dotenv
from output_models.obj_design_structre import *

load_dotenv()  # Load environment variables from .env


# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = openai_api_key
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

def read_context(file_path):
    with open(file_path, 'r') as file:
        context = file.read()
    return context

root_path = "/home/malineni/PycharmProjects/SayCan"
context_file = root_path + "/resources/llm_contexts/llm_action_designators.txt"
context = read_context(context_file)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/plan', methods=['GET', 'POST'])
def planner():
    if request.method == 'POST':
        prompt = request.form['prompt']

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that translates natural language instructions into Python code for robotic action plan for pr2 robot."},
                {"role": "assistant", "content": context},
                {"role": "user", "content": f"resolve the natural langugae instruction {prompt} into a python code of action designators from the context"},
                {"role": "user", "content": "plan navigation for robot if needed and Output only python code"}
            ],
            max_tokens=1000,
            temperature=0
            # messages=[
            #     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            #     {"role": "user", "content": prompt}
            # ]
        )

        code = response.choices[0].message.content.strip()

        with open(root_path + "/outputs/gen_code.py", "w") as file:
            file.write(code)

        return render_template('home.html', prompt=prompt, response=response.choices[0].message.content)
    else:
        return render_template('home.html')

@app.route('/segment', methods=['GET', 'POST'])
def resolve_objects():
    if request.method == 'POST':
        prompt = request.form['prompt']

        env = ("{'floor': {'name': 'floor', 'obj_type': <ObjectType.ENVIRONMENT: 9>}, "
               "'kitchen': {'name': 'kitchen', 'obj_type': <ObjectType.ENVIRONMENT: 9>}, "
               "'milk': {'name': 'milk', 'obj_type': <ObjectType.MILK: 3>}, "
               "'froot_loops': {'name': 'froot_loops', 'obj_type': <ObjectType.BREAKFAST_CEREAL: 6>}, "
               "'spoon': {'name': 'spoon', 'obj_type': <ObjectType.SPOON: 4>}}")

        env_context = (f"Objects in the environment and their names, types are as follows "
                       f"{env}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that can detect objects in prompt and "
                            "match semantically to objects in environment context"},
                {"role": "assistant", "content": env_context},
                {"role": "user", "content": f"find all the objects in the prompt {prompt} that exactly matches"
                                            f"with the objects in environment context and "
                                            f"print their names only no explanation needed"},
                {"role": "user", "content": "mention 'No Such Object in the Environment' if nothing matches"}
            ],
            max_tokens=1000,
            temperature=0
        )

        # code = response.choices[0].message.content.strip()

        # with open("/home/malineni/ROS_WS/tpycram_ws/src/pycram/src/pycram/llms/outputs/gen_code.py", "w") as file:
        #     file.write(code)

        return render_template('segment_home.html', prompt=prompt, response=response.choices[0].message.content)
    else:

        return render_template('segment_home.html')

# @app.route('/segment', methods=['GET', 'POST'])
# def decompose_disambiguate():
#     if request.method == 'POST':
#         prompt = request.form['prompt']
#
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system",
#                  "content": "You are a helpful assistant that separate high level natural language instructions into simpler instructions suitable for robots"},
#                 # {"role": "assistant", "content": context},
#                 {"role": "user", "content": f"resolve the natural language instruction {prompt} into separate meaningful sentences"},
#                 # {"role": "user", "content": "plan navigation for robot if needed and Output only python code"}
#             ],
#             max_tokens=1000,
#             temperature=0
#             # messages=[
#             #     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#             #     {"role": "user", "content": prompt}
#             # ]
#         )
#
#         # code = response.choices[0].message.content.strip()
#
#         # with open("/home/malineni/ROS_WS/tpycram_ws/src/pycram/src/pycram/llms/outputs/gen_code.py", "w") as file:
#         #     file.write(code)
#
#         return render_template('segment_home.html', prompt=prompt, response=response.choices[0].message.content)
#     else:
#         return render_template('segment_home.html')

@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    if request.method == 'POST':
        prompt = request.form['prompt']

        obj_context_file = root_path + "/resources/llm_contexts/llm_object_designators.txt"
        obj_context = read_context(obj_context_file)


        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that can detect different objects in the instruction and"
                            "find relevant object designators for each from the context"},
                {"role": "assistant", "content": obj_context},
                {"role": "user",
                 "content": f"resolve different objects in {prompt} and match best suited object designator one for each"},
                {"role": "user",
                 "content": "Only output object designators instances with resolved parameters, no explanation needed"}
            ],
            max_tokens=1000,
            temperature=0,
            # response_format =
        )

        # res = response.choices[0].message.content
        # json_res = json.loads(res)
        # print("MODEL RESPONSE : ", res, type(res), type(json_res))
        # print(f"ObjectDesignatorDescription(names={json_res['names']}, types={json_res['types']})")
        # print(f"ObjectDesignatorDescription(names={response.choices[0].message.content["names"]})")
        print(type(response.choices[0].message.content.split('\n')))
        queried_obj_designators =  response.choices[0].message.content.split('\n')
        print("queried_obj_designators : ", queried_obj_designators)

        # obj_context = f"Object designators of queried objects are \n\n {queried_obj_designators}"
        # act_context_file = root_path + "/resources/llm_contexts/llm_action_designators.txt"
        # act_context = read_context(act_context_file)
        # act_context = act_context + "\n\n" + obj_context + "\n\n" + f"PROMPT = {prompt}"
        #
        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system",
        #          "content": "You are a helpful assistant that can map the prompt and object designators "
        #                     "from the context into best suited action designators from the context"},
        #         {"role": "assistant", "content": act_context},
        #         {"role": "user", "content": f"resolve the prompt {prompt} into best "
        #                                     f"suited action designators, not action performables, from the context"},
        #         {"role": "user", "content": "Only output action designators with resolved parameters, no explanation needed"}
        #     ],
        #     max_tokens=1000,
        #     temperature=0
        # )

        return render_template('classifier_home.html', prompt=prompt, response=response.choices[0].message.content)

    else:
        return render_template('classifier_home.html')

if __name__ == '__main__':
    app.run(debug=True)