from openai import OpenAI
import re
import os
import json
from .prompt import *
from .api import openai_send_messages
from concurrent.futures import ThreadPoolExecutor, as_completed

class MASecDev:
    def __init__(self):
        self.user_msg = ""
        self.prd = ""
        self.call_chain = ""
        self.arch = {"files": [] }
        self.files = {}
        self.code = []
        self.threat_model = ""
        self.work_dir = "./ma-secdev-test-workspace"
        self.threat_model_mode = "simple"  # or "call_chain"
        
        
        
    def write_prd(self, websearch=False):
        if not websearch:
            input_prompt = WRITE_PRD_PROMPT.format(msg=self.user_msg)
            response = openai_send_messages(input_prompt)
            print(response)
            self.prd = response
    
    def write_system_design(self):
        
        ## 先设计一下所需要的文件架构
        file_design_prompt = WRITE_FILE_DESIGN_PROMPT.format(msg=self.user_msg, prd=self.prd)
        response = openai_send_messages(file_design_prompt)
        print(response)
        try:
            matches = re.search(r'(\{.*\})', response, re.DOTALL)
            json_str = matches.group(1)
            self.files = json.loads(json_str)
        except Exception as e:
            raise ValueError("文件设计生成的JSON格式有误，请检查输出内容。错误详情：" + str(e))
        
        for file in self.files['files']:
            # 设计每个文件的具体内容如类和函数定义
            input_prompt = WRITE_SYSTEM_DESIGN_PROMPT.format(msg=self.user_msg, 
                                                             prd=self.prd, 
                                                             api=json.dumps(self.arch["files"][-5:], indent=2),
                                                             files=json.dumps(self.files, indent=2),
                                                             file_name = (
                                                                f"### File Name:{file['name']}\n"
                                                                f"### File Path:{file['path']}"
                                                            ))
            
            print(input_prompt)
            response = openai_send_messages(input_prompt)
            print(response)
            try:
                matches = re.search(r'(\{.*\})', response, re.DOTALL)
                json_str = matches.group(1)
                next_file = json.loads(json_str)
                self.arch['files'].append(next_file)
                
            except Exception as e:
                raise ValueError("系统设计生成的JSON格式有误，请检查输出内容。错误详情：" + str(e))

         
        
    
    def write_threat_modeling(self):
        
        def build_threat_model_for_interface(interface):
            prompt = THREAT_MODELING_PER_CHAIN_PROMPT.format(msg=self.user_msg, api=self.arch, prd=self.prd, interface=json.dumps(interface, indent=2))
            response = openai_send_messages(prompt)
            return response
        
        
        if self.threat_model_mode == "simple":
            input_prompt = THREAT_MODELING_PROMPT.format(msg=self.user_msg, api=self.arch)
            response = openai_send_messages(input_prompt)
            print(response)
            self.threat_model = response
        
        elif self.threat_model_mode == "call_chain":
            tm_list = []
            for _attempt in range(3):
                get_call_chain_prompt = GET_CALL_CHAIN_PROMPT.format(api=self.arch)
                call_chain_response = openai_send_messages(get_call_chain_prompt)
                print(call_chain_response)
                self.call_chain = call_chain_response
                json_str = re.search(r'(\{.*\})', call_chain_response, re.DOTALL).group(1)
                interfaces = json.loads(json_str)['interfaces']
                threat_models = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(build_threat_model_for_interface, interface)
                        for interface in interfaces
                    ]

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            threat_models.append(result)
                        except Exception as e:
                            print(f"Error generating threat model for an interface: {e}")
                # 合并所有接口的威胁模型
                combined_threat_model_prompt = COMBINE_THREAT_MODELS_PROMPT.format(
                    msg=self.user_msg,
                    api=self.arch,
                    threat_models="\n\n".join(threat_models)
                )
                combined_response = openai_send_messages(combined_threat_model_prompt)
                print(combined_response)
                tm_list.append(combined_response)
            
            tm_scores = []
            for i, tm in enumerate(tm_list):
                print(f"=== 威胁模型方案 {i+1} ===")
                print(tm)
                # 选择最好的威胁模型作为最终结果
                judge_prompt = JUDGE_THREAT_MODELS_PROMPT.format(
                    msg=self.user_msg,
                    api=self.arch,
                    threat_models=tm
                )
                
                ## 计算总分数，选择最高的
                score = openai_send_messages(judge_prompt)
                
                json_score = re.search(r'(\{.*\})', score, re.DOTALL).group(1)
                score_dict = json.loads(json_score)['output']
                total_score = score_dict['relevance'] + score_dict['impact'] + score_dict['exploitability']
                tm_scores.append((total_score, tm))
            # 选择最高分的威胁模型
            tm_scores.sort(key=lambda x: x[0], reverse=True)
            self.threat_model = tm_scores[0][1]
            

    def write_code(self):

        files = self.arch['files']

        for f in files:
            file_path = os.path.join(self.work_dir, f["path"], f["name"])

            # 如果文件存在，跳过
            if os.path.exists(file_path):
                print(f"[skip] 文件已存在：{file_path}")
                continue
            
            # ==== 使用你提供的 ENGINEER_PROMPT ====
            input_prompt = ENGINEER_PROMPT.format(
                msg=self.user_msg,
                api=self.arch,
                threat=self.threat_model,
                code='\n\n'.join(self.code),   # 关键：给 GPT 10个依赖代码
                file_name=os.path.join(f["path"], f["name"]),
            )

            response = openai_send_messages(input_prompt)
            print(response)

            # 聚合新代码
            self.code.append(response)

            # 写文件
            self._save(response)

        
    def run(self, user_msg, begin_stage="prd"):
        self.user_msg = user_msg

        # 加载之前可能已有的 prd / arch / threat 文件
        self.load_existing_state()

        stage_order = ["prd", "system_design", "threat_model", "code", "readme"]

        # 找到 begin_stage 的位置
        if begin_stage not in stage_order:
            raise ValueError(f"Invalid begin_stage: {begin_stage}")

        start_index = stage_order.index(begin_stage)

        # 从 begin_stage 往后依次执行
        for stage in stage_order[start_index:]:
            if stage == "prd":
                self.write_prd()
            elif stage == "system_design":
                # 依赖 prd, 所以必须 prd 不为空
                if not self.prd:
                    raise RuntimeError("PRD is empty but system_design requested")
                self.write_system_design()
            elif stage == "threat_model":
                if not self.arch:
                    raise RuntimeError("Architecture is empty but threat_model requested")
                self.write_threat_modeling()
            elif stage == "code":
                if not self.threat_model:
                    raise RuntimeError("Threat_model is empty but code generation requested")
                self.write_code()
            elif stage == "readme":
                self.write_readme()
    
    
    def write_readme(self):
        """
        write a README.md file based on the generated code files.
        """
        import os

        # make sure work_dir exists
        print(f"creating work dir: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        readme_content = openai_send_messages(WRITE_README_PROMPT.format(code='\n\n'.join(self.code)), temperature=0.3)

        # write to README.md
        readme_path = os.path.join(self.work_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        self.readme = readme_content
        print(f"Generated README: {readme_path}")

    
    def _save(self, code):
        import os
        import re

        text = code

        blocks = re.findall(r"<file>(.*?)</file>", text, flags=re.DOTALL)


        print(f"创建工作目录: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        written_files = []

        block = blocks[0]
        # 先去掉首尾空白
        b = block.strip()

        # 分割成行，方便按行处理
        lines = [line.rstrip("\n") for line in b.splitlines()]

        file_path = None
        code_lines = []

        # 状态机：目前是否在 Code 段落内
        in_code = False

        for i, line in enumerate(lines):
            # 找 "## File"
            if line.strip().startswith("## File"):
                # 下方第一行非空即为路径
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines):
                    file_path = lines[j].strip()

            # 找 "## Code"
            if line.strip().startswith("## Code"):
                # 之后所有内容都当成代码
                in_code = True
                # 跳过这一行本身
                continue

            if in_code:
                code_lines.append(line)

        if not file_path:
            raise ValueError("警告：某个 <file> 块未找到文件路径，已跳过")
            

        code = "\n".join(code_lines).lstrip("\n")

        # 防止绝对路径 / 越权（简单版）
        file_path = file_path.lstrip("/\\")

        full_path = os.path.join(self.work_dir, file_path)
        dir_name = os.path.dirname(full_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code)

        written_files.append(full_path)
        print(f"已写入文件: {full_path}")

        if not written_files:
            raise ValueError("解析成功但没有任何文件被写入，请检查输出格式是否正确")

        # === 在工作目录根目录下写入 prd / arch / threat_model 三个 txt 文件 ===
        meta_files = {
            "prd.txt": self.prd,
            "arch.txt": json.dumps(self.arch, indent=4),
            "threat_model.txt": self.threat_model,
        }
        for fname, content in meta_files.items():
            meta_path = os.path.join(self.work_dir, fname)
            if os.path.exists(meta_path):
                print(f"元信息文件已存在，跳过写入: {meta_path}")
                continue
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(content or "")
            print(f"已写入元信息文件: {meta_path}")

        return written_files
    
    
    def load_existing_state(self):
        prd_path = os.path.join(self.work_dir, "prd.txt")
        arch_path = os.path.join(self.work_dir, "arch.txt")
        threat_path = os.path.join(self.work_dir, "threat_model.txt")

        if os.path.exists(prd_path):
            with open(prd_path, "r", encoding="utf-8") as f:
                self.prd = f.read()

        if os.path.exists(arch_path):
            with open(arch_path, "r", encoding="utf-8") as f:
                self.arch = json.loads(f.read())

        if os.path.exists(threat_path):
            with open(threat_path, "r", encoding="utf-8") as f:
                self.threat_model = f.read()

        # NEW：格式化已有代码
        self.code = self.format_existing_code()



    def format_existing_code(self):
        """Return all existing code files formatted in <file>...</file> blocks."""
        formatted = []

        EXCLUDE_DIRS = {
            "node_modules", ".git", "__pycache__",
            "dist", "build", ".venv", "venv"
        }

        for root, dirs, files in os.walk(self.work_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                if not file.endswith((".py", ".js", ".php", ".ts", ".html", ".go", ".java", ".c", ".cpp")):
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.work_dir)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code_content = f.read()
                except Exception:
                    continue  # 跳过读不了的文件

                block = f"""
<file>
## File
{rel_path}

## Code
{code_content}
</file>"""

                formatted.append(block)

        return formatted