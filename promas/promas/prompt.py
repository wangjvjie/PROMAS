

WRITE_PRD_PROMPT = """
You are a project manager, the user will give an instruction for coding, your job is to understand the requirements and come up with the Implementation approach.
You first need to understand the user’s underlying needs and refine them accordingly.
Second, for the current task, you should consider security considerations at both the macro level and in practical deployment, including measures that can be implemented programmatically. For example, using image-based CAPTCHAs to prevent malicious registrations, and setting up anti-scraping mechanisms.
Finally, you need to take into account all possible configuration files and related settings to ensure that the user can run the code with a single command.

You should response in the following format:

```
## User Story
xxx

## System-Level Security Design
XXX

## Implementation approach
xxx.
```
[Example]
User input: I want to write a python3 GUI app such that you can draw an image with it.

Your output:
```
## User Story
As a user, I want a simple python3 GUI application that allows me to draw images using mouse input. The application should provide basic drawing functionalities such as freehand drawing, color selection, and the ability to save/export the drawn image.

## System-Level Security Design
XXX

## Implementation approach
The user want a python3 GUI app for drawing image, the implementation approach are as follows:
- Use Tkinter for GUI (built into Python).
- Create a canvas widget for drawing.
- Bind mouse events to enable free drawing.
- Add save/export button to store the image.

## config files
- requirements.txt including any external libraries (if needed).
- vite.config.js for frontend setup (if using Vite).
- twailwind.config.js for Tailwind CSS setup (if using Tailwind).
- tsconfig.json for TypeScript setup (if using TypeScript).
- package.json for Node.js dependencies (if using Node.js).
- index.html as the main HTML file.
- tsconfig.node.json for Node.js TypeScript setup (if using TypeScript with Node.js).
....

```
You only need to output the content in the above format. Do not output any additional explanatory text, and you do not need to write any code here.


[User Messages]
{msg}
"""



WRITE_FILE_DESIGN_PROMPT = """
You are a senior system architect.

Your task is to DESIGN the file structure and internal APIs for a small system, but DO NOT write any code yet.
Note that at this step, you must generate all configuration files required by the scaffolding you use, because it is necessary to ensure one-click execution in the end. For example, if the front end uses Vite, you must generate files such as vite.config.js. You must not omit any required configuration files.
In addition, the goal of the code you will eventually write is that it can be run in one go, so please ensure that the code is complete and runnable. The user should be able to run the code with a single command, such as php -S or npm run dev.
The goal of this step is to design all required file names and paths. Please output these file names and paths in the form of a JSON file.
In addition, you need to consider common industry protection measures, such as image-based or slider CAPTCHAs for human verification, anti-scraping mechanisms, behavior and rate limiting protections, and so on.

## User Message
{msg}

## User Story
{prd}

Return the FULL FILE DESIGN in valid JSON with the following schema:
{{
  "files": [
    {{
      "name": "string.py (c, cpp, etc)",
      "path": "./xxx (If you think the file should be placed in the root directory, output ./)",
      "description": "string"
    }},
    ...
  ]
}}

"""


WRITE_SYSTEM_DESIGN_PROMPT = """
You are a senior system architect.

Your task is to DESIGN the file structure and internal APIs for a small system, but DO NOT write any code yet.
Your task is to complete the current file design and define the classes and functions for that file.


## User Message
{msg}

## User Story
{prd}

## All file structures
{files}

## History System Design
{api}

Return the File Design in valid JSON with the following schema:
{{
  "name": "string.py (c, cpp, etc)",
  "path": "./xxx (If you think the file should be placed in the root directory, output ./)",
  "classes": [
      {{
          "class_name": "string"
          "members": [
              {{
                  "name": "string",
                  "type": "str, boolean, etc"
              }}
          ]
      }}
  ]
  "functions": [
    {{
      "name": "class.funcname (If you think the function doesn't need a class, just output funcname)",
      "input_parameters": [
          {{
              "name": "<name>"
              "type": "<type>"
          }}
      ],
      "output_parameters": [
          {{
              "name": "<name>"
              "type": "<type>"
          }}
      ],
    }}
  ]
}}

Only output this JSON. Do NOT write any implementation code.

## The file you are currently designing is
{file_name}

Please output the design of this file:

"""
 


THREAT_MODELING_PROMPT = """
Given the following system architecture and functions, produce a concise, task-specific THREAT MODEL ONLY.
Do NOT write any code. Stick to the sections and bullet points. Be concrete to THIS task.

You need to assess the potential security threats present in each function within the architecture based on the code architecture and functionality I provide, and construct a threat model.
Note: The architecture may contain multiple functions. You need to analyze them one by one, and then provide an overall summary.

Please respond in the following format (Markdown):

## 1. Global Security Context
- Overall purpose:
- Main trust boundaries (untrusted inputs, external deps):
- Key assets & privileges touched:

## 2. Function-level Threats
Note: The potential threats should be as comprehensive as possible, covering all possible threats.
### Function: <path/file:function_name>
- Role in the system:
- Untrusted inputs / external sources:
- Security-relevant operations (e.g., file I/O, DB, network, eval/exec, template rendering):
- Potential threats:
  - T1: <short name> — <1–2 line description>
  - T2: ...
- Recommended protections (high-level):

### Function: <path/file:function_name>
- Role in the system:
- Untrusted inputs / external sources:
- Security-relevant operations:
- Potential threats:
  - ...
- Recommended protections:

(Repeat the above block for EACH function in the given architecture.)


## 3. Attacker Model
- Capabilities: Control `target` (scheme/host/port/userinfo/path/query/fragment); try Unicode/IDN, mixed case, dot tricks.
- Goals: Open redirect; egress to attacker domain; downgrade to http; SSRF via special hosts (e.g., 127.0.0.1).`


Requirements:
- Do not output any code or pseudocode at any point; only output threat analysis.
- In the “Function: ...” headings, you must clearly specify the function name (preferably including the file name, e.g., backend/app.py:list_posts).
- For “Potential threats,” use concise bullet points that focus on security risks directly related to this task.


Below are the given APIs and user messages
## User Message
{msg}

## System Design
{api}

"""



ENGINEER_PROMPT = """
You are an experienced programmer, and your goal is to write secure and fully functional code. I will provide you with a set of user goals, along with a designed API architecture and threat model.
In addition, the goal of the code you write is that it can be run in one go, so please ensure that the code is complete and executable.

## User Message
{msg}

## System Design
{api}

## Threat Model
{threat}


## Already written code
{code}

The file name you are currently going to write: {file_name}

Please output in the following format:
[Output Format]
<file>
## File
path/file_name

## Code
All the code for this file (write code only, do not include Markdown formatting, and do not include any extra explanatory text)
</file>
[/Output Format]

Your output:
"""

WRITE_README_PROMPT = """
You are an experienced software engineer and technical writer.

I will give you the full system design of a small project.
write a clear and concise README.md that explains how to set up and run the project.

Requirements:
- Use Markdown format, but DO NOT wrap the whole content in ``` code fences.
- Infer the programming language(s), framework(s), and main entry point from the code.
- If it's a web service, explain how to start the server and which URL to open.
- If there are external dependencies (e.g., Flask, requests), specify how to install them (e.g., pip install ...).
- Include at least the following sections:
  - Introduction / Overview
  - Requirements
  - Installation
  - How to Run
  - (Optional) Examples or Usage
- If something is ambiguous, make a reasonable assumption and state it briefly.

Here is the full project code:

{code}
"""

GET_CALL_CHAIN_PROMPT = """
You are a senior system architect.
Your task is to extract all interfaces and their call chain functions from the System Design

## System Design
{api}

You should response in the following json format:
```json
{ 
  "interface": [{
    "funcname": "str(用户可能的入口函数名称, 格式为file_path/file_name:(可选class.)funcname)",
      "call-chain": [
      "callee_func-1-name(格式同样为file_path/file_name:(可选class.)funcname)",
      "callee_func-2-name"
    ]},
    {
    "funcname": "str(用户可能的入口函数名称)",
      "call-chain": [
      "callee_func-1-name",
      "callee_func-4-name"
    ]
}
请你只输出json, 无需其他任何解释文字。
"""


THREAT_MODELING_PER_CHAIN_PROMPT = """
# Task
Given the system architecture and a specific **Function Call Chain**, produce a concise, task-specific **THREAT MODEL**. You must perform a step-by-step security analysis for **EACH** function in the provided chain, focusing on how untrusted data propagates and where security controls might fail.

# Constraints
1. Do NOT write any code.
2. Stick to the sections and bullet points provided below.
3. BE CONCRETE: Do not use generic security advice. Reference the specific logic described in the System Design.
4. COMPREHENSIVENESS: Cover all possible threats including but not limited to Injection, Broken Auth, Data Exposure, and Supply Chain risks.

# Input Data
## User Message
{msg}

## User Story
{prd}

## Architecture Design
{api}

## Target Call Chain to Analyze
{interface}

# Output Format

## Function-level Threats

### Function: <path/file:function_name>
- **Role in the system**: [Brief description of the function's responsibility]
- **Untrusted inputs / external sources**: [List any parameters or external data this function consumes]
- **Security-relevant operations**: [e.g., file I/O, DB, network, eval/exec, template rendering, encryption]
- **Potential threats**:
  - **T1: <short name>** — <1–2 line description of the specific attack vector>
  - **T2: <short name>** — <1–2 line description>
- **Recommended protections**: [High-level mitigation strategies, e.g., input sanitization, parameterized queries]

---
*(Repeat the above block for EACH function in the given call chain)*
"""


COMBINE_THREAT_MODELS_PROMPT = """
Please combine the following multiple threat models into a complete threat model.
Please output in the following format:

## User Message
{msg}

## System Design
{api}

## Target Threat Model
{threat_models}

Please output the final threat model in the following format:
## 1. Global Security Context
- Overall purpose:
- Main trust boundaries (untrusted inputs, external deps):
- Key assets & privileges touched:

## 2. Function-level Threats
Note: The potential threats should be as comprehensive as possible, covering all possible threats.
### Function: <path/file:function_name>
- Role in the system:
- Untrusted inputs / external sources:
- Security-relevant operations (e.g., file I/O, DB, network, eval/exec, template rendering):
- Potential threats:
  - T1: <short name> — <1–2 line description>
  - T2: ...
- Recommended protections (high-level):

### Function: <path/file:function_name>
- Role in the system:
- Untrusted inputs / external sources:
- Security-relevant operations:
- Potential threats:
  - ...
- Recommended protections:

(Repeat the above block for EACH function in the given architecture.)


## 3. Attacker Model
- Capabilities: Control `target` (scheme/host/port/userinfo/path/query/fragment); try Unicode/IDN, mixed case, dot tricks.
- Goals: Open redirect; egress to attacker domain; downgrade to http; SSRF via special hosts (e.g., 127.0.0.1).`


Requirements:
- Do not output any code or pseudocode at any point; only output threat analysis.
- In the “Function: ...” headings, you must clearly specify the function name (preferably including the file name, e.g., backend/app.py:list_posts).
- For “Potential threats,” use concise bullet points that focus on security risks directly related to this task.


"""

JUDGE_THREAT_MODELS_PROMPT = """
Please evaluate the given security threats based on the following API interface design and user requirements:

1. **Relevance (1–5):** Measures how relevant the threat modeling is to the project.
2. **Impact (1–5):** Evaluates how significant the security threats mentioned are.
3. **Exploitability (1–5):** Assesses how easily the security threats described in the threat modeling can be exploited by attackers.

## Output Format (JSON)

```json
{
  "output": {
    "relevance": <1-5>,
    "impact": <1-5>,
    "exploitability": <1-5>
  }
}
```



# System Design
{api}

# User Message
{msg}


# Threat Model
{threat}
"""