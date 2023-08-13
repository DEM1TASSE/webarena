import os
import openai
import json
import re
import subprocess
import time

SLEEP = 1
# set the URLs of each website, we use the demo sites as an example
os.environ[
    "SHOPPING"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ[
    "SHOPPING_ADMIN"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
os.environ[
    "REDDIT"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
os.environ[
    "GITLAB"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
os.environ[
    "MAP"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ[
    "WIKIPEDIA"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site
print("Done setting up URLs")

from agent.utils import Trajectory

# Init an environment
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    ScriptBrowserEnv,
    StateInfo,
    action2str,
    create_id_based_action,
    create_stop_action,
)
from evaluation_harness.evaluators import evaluator_router

# Init the environment
env = ScriptBrowserEnv(
    headless=False,
    slow_mo=100,
    observation_type="accessibility_tree",
    current_viewport_only=True,
    viewport_size={"width": 1280, "height": 720},
)

config_file = "config_files/164.json"

trajectory: Trajectory = []
obs, info = env.reset(options={"config_file": config_file})
actree_obs = obs["text"]

openai.api_key = "OPENAI_API_KEY"

observation = ""
previous_action = []
memory = {}

messages=[
    {
        "role": "system",
        "content": "You are an autonomous intelligent agent that is great at website design, navigation, and executing tasks for the user. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue. "
    },
]
instruction = "Instruction:\nMemory:\nIn this task, you have a short-term memory in JSON format. You can add useful information that you consider relevant to the memory in JSON format. At each step, you need to reason based on the current observation and the previous memory. Determine whether the current observation contains valuable information, thus deciding whether to update or retain the memory. Record your reasoning process in the 'memory_reasoning' section.\n\nAction:\nIn this task, at each step, you should reason based on the current observation and the current memory. Predict the most likely operation to accomplish the task in the next step. Document your reasoning process in the 'action_reasoning' section.\n\nThe actions you can perform fall into several categories:\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [direction=down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\nIf you believe the task is completed, use \"stop [answer]\" to stop actions and submit the final answer.\n\nHere is a task you need to complete:\n\"task\": \"What are the main criticisms of this product? Please extract the relevant sentences."
hint = "\nHints:\n1. 首先检查当前observation中有没有评论相关的内容，如果有则加入memory。评论内容示例：[2576] StaticText 'Those labial never came to me and I would like them again because I paid for them'。\n2. 然后判断memory中评论相关的内容是否被提取完全，如果提取完全，则下一步action是：用stop [answer]提交答案。\n3. 检查previous_action，提取里面的操作。\n4. 你可以尝试点击所有和reviews相关的元素，但不要对previous_action的元素重复操作。\n5. 如果页面中没有你想要的信息，你可以scroll [down]。\n\nReason based on the current observation, memory, and previous action to determine whether updating the memory is necessary to achieve the goal. Predict the next step's action and represent it in the specified format.\nPresent your answer in JSON format, including 'memory_reasoning', 'memory', 'action', and 'action_reasoning', . Do not include  \"observation\" and \"task\" in your answer."

cnt = 0

def format_action(current_action):
    pattern = r"([a-zA-Z]+)\s+(\[?\d+\]?)"
    match = re.match(pattern, current_action)
    
    if match:
        action = match.group(1)
        number = match.group(2)
        
        if not number.startswith("[") and not number.endswith("]"):
            number = f"[{number}]"
        
        formatted_action = f"{action} {number}"
        return formatted_action
    else:
        return current_action

while True:
    prompt = instruction + "\n\"observation\":" + actree_obs + "\n" + str(memory) + "\n\"previous_action\"" + str(previous_action) + "\n" + hint

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an autonomous intelligent agent that is great at website design, navigation, and executing tasks for the user. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue. "
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print("response", response)

    # Extract memory and action
    chat_response_text = response.choices[0]['message']['content']
    print("chat_response_text", chat_response_text)
    chat_response = json.loads(chat_response_text)
    current_memory = chat_response.get('memory', {})
    current_action = chat_response.get('action', '')
    current_action = format_action(current_action)
    print("current_memory", current_memory)
    print("current_action", current_action)

    # # 检查当前操作是否与previous_action重复
    # while current_action in previous_action and current_action != "scroll [down]":
    #     repeat_count += 1
    #     if repeat_count >= 3:
    #         print("重复次数超过3次，终止循环。")
    #         break
    #     print("检测到重复操作，重新预测。")
    #     # 此处可以再次调用openai API进行重新预测
    #     # ...

    # check current action is stop
    if "stop" in current_action:
        # action 5 stop action for result
        action = create_stop_action(current_action)
        trajectory.append(action)
        print("Stop action", action)
        break

    # update memory and previous_action
    memory.update(current_memory)
    previous_action.append(current_action)

    # execute action
    action = create_id_based_action(current_action)
    trajectory.append(action)
    print("action:", action)
    obs, _, terminated, _, info = env.step(action)
    actree_obs = obs["text"]
    print("observation:", actree_obs)
    time.sleep(SLEEP)
    state_info = {"observation": obs, "info": info}
    trajectory.append(state_info)

    cnt += 1
    if cnt >= 10:
        break