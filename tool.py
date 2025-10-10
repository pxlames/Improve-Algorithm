"""
基于LangChain的工具集合
包含文件操作、网络请求、数据处理、文本处理、数学计算、日期时间等多种实用工具
"""

from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Optional, List, Dict, Any, Union
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import hashlib
import base64
import zipfile
import shutil
from pathlib import Path
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 文件操作工具 ====================

class FileReadInput(BaseModel):
    """文件读取工具输入参数"""
    file_path: str = Field(description="要读取的文件路径")
    encoding: str = Field(default="utf-8", description="文件编码格式")


class FileReadTool(BaseTool):
    """文件读取工具"""
    name = "file_read"
    description = "读取指定路径的文件内容"
    args_schema = FileReadInput

    def _run(self, file_path: str, encoding: str = "utf-8") -> str:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return f"文件 {file_path} 读取成功，内容长度: {len(content)} 字符"
        except Exception as e:
            return f"读取文件失败: {str(e)}"


class FileWriteInput(BaseModel):
    """文件写入工具输入参数"""
    file_path: str = Field(description="要写入的文件路径")
    content: str = Field(description="要写入的内容")
    encoding: str = Field(default="utf-8", description="文件编码格式")


class FileWriteTool(BaseTool):
    """文件写入工具"""
    name = "file_write"
    description = "将内容写入指定路径的文件"
    args_schema = FileWriteInput

    def _run(self, file_path: str, content: str, encoding: str = "utf-8") -> str:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding=encoding) as file:
                file.write(content)
            return f"文件 {file_path} 写入成功，内容长度: {len(content)} 字符"
        except Exception as e:
            return f"写入文件失败: {str(e)}"


class FileListInput(BaseModel):
    """文件列表工具输入参数"""
    directory_path: str = Field(description="要列出文件的目录路径")
    pattern: Optional[str] = Field(default=None, description="文件匹配模式")


class FileListTool(BaseTool):
    """文件列表工具"""
    name = "file_list"
    description = "列出指定目录下的文件"
    args_schema = FileListInput

    def _run(self, directory_path: str, pattern: Optional[str] = None) -> str:
        try:
            path = Path(directory_path)
            if not path.exists():
                return f"目录 {directory_path} 不存在"
            
            files = []
            if pattern:
                files = list(path.glob(pattern))
            else:
                files = list(path.iterdir())
            
            file_info = []
            for file in files:
                if file.is_file():
                    size = file.stat().st_size
                    file_info.append(f"{file.name} ({size} bytes)")
                elif file.is_dir():
                    file_info.append(f"{file.name}/ (目录)")
            
            return f"目录 {directory_path} 下的文件:\n" + "\n".join(file_info)
        except Exception as e:
            return f"列出文件失败: {str(e)}"


# ==================== 网络请求工具 ====================

class HttpRequestInput(BaseModel):
    """HTTP请求工具输入参数"""
    url: str = Field(description="请求的URL")
    method: str = Field(default="GET", description="HTTP方法 (GET, POST, PUT, DELETE)")
    headers: Optional[Dict[str, str]] = Field(default=None, description="请求头")
    data: Optional[Dict[str, Any]] = Field(default=None, description="请求数据")
    timeout: int = Field(default=30, description="请求超时时间(秒)")


class HttpRequestTool(BaseTool):
    """HTTP请求工具"""
    name = "http_request"
    description = "发送HTTP请求并获取响应"
    args_schema = HttpRequestInput

    def _run(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, 
             data: Optional[Dict[str, Any]] = None, timeout: int = 30) -> str:
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=headers,
                json=data,
                timeout=timeout
            )
            return f"状态码: {response.status_code}\n响应内容: {response.text[:1000]}..."
        except Exception as e:
            return f"HTTP请求失败: {str(e)}"


# ==================== 数据处理工具 ====================

class DataProcessInput(BaseModel):
    """数据处理工具输入参数"""
    data: str = Field(description="要处理的数据(JSON格式)")
    operation: str = Field(description="操作类型: sort, filter, group, aggregate")


class DataProcessTool(BaseTool):
    """数据处理工具"""
    name = "data_process"
    description = "处理JSON数据，支持排序、过滤、分组、聚合等操作"
    args_schema = DataProcessInput

    def _run(self, data: str, operation: str) -> str:
        try:
            data_obj = json.loads(data)
            
            if operation == "sort":
                if isinstance(data_obj, list):
                    sorted_data = sorted(data_obj)
                    return f"排序结果: {json.dumps(sorted_data, ensure_ascii=False)}"
            
            elif operation == "filter":
                # 简单的过滤示例
                if isinstance(data_obj, list):
                    filtered = [item for item in data_obj if isinstance(item, (int, float)) and item > 0]
                    return f"过滤结果: {json.dumps(filtered, ensure_ascii=False)}"
            
            return f"操作 {operation} 执行完成"
        except Exception as e:
            return f"数据处理失败: {str(e)}"


# ==================== 文本处理工具 ====================

class TextProcessInput(BaseModel):
    """文本处理工具输入参数"""
    text: str = Field(description="要处理的文本")
    operation: str = Field(description="操作类型: clean, extract_email, extract_phone, word_count, hash")


class TextProcessTool(BaseTool):
    """文本处理工具"""
    name = "text_process"
    description = "处理文本，支持清理、提取邮箱/电话、词数统计、哈希等操作"
    args_schema = TextProcessInput

    def _run(self, text: str, operation: str) -> str:
        try:
            if operation == "clean":
                # 清理文本
                cleaned = re.sub(r'\s+', ' ', text.strip())
                return f"清理后的文本: {cleaned}"
            
            elif operation == "extract_email":
                # 提取邮箱
                emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
                return f"提取到的邮箱: {emails}"
            
            elif operation == "extract_phone":
                # 提取电话号码
                phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
                return f"提取到的电话号码: {phones}"
            
            elif operation == "word_count":
                # 词数统计
                words = len(text.split())
                chars = len(text)
                return f"词数: {words}, 字符数: {chars}"
            
            elif operation == "hash":
                # 生成哈希值
                hash_md5 = hashlib.md5(text.encode()).hexdigest()
                hash_sha256 = hashlib.sha256(text.encode()).hexdigest()
                return f"MD5: {hash_md5}\nSHA256: {hash_sha256}"
            
            return f"操作 {operation} 执行完成"
        except Exception as e:
            return f"文本处理失败: {str(e)}"


# ==================== 数学计算工具 ====================

class MathCalcInput(BaseModel):
    """数学计算工具输入参数"""
    expression: str = Field(description="数学表达式")
    variables: Optional[Dict[str, float]] = Field(default=None, description="变量值")


class MathCalcTool(BaseTool):
    """数学计算工具"""
    name = "math_calc"
    description = "计算数学表达式，支持基本运算和常用函数"
    args_schema = MathCalcInput

    def _run(self, expression: str, variables: Optional[Dict[str, float]] = None) -> str:
        try:
            # 安全的数学表达式计算
            allowed_names = {
                k: v for k, v in __builtins__.items()
                if k in ['abs', 'min', 'max', 'round', 'sum']
            }
            allowed_names.update({
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'log': np.log, 'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e
            })
            
            if variables:
                allowed_names.update(variables)
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"计算结果: {result}"
        except Exception as e:
            return f"数学计算失败: {str(e)}"


# ==================== 日期时间工具 ====================

class DateTimeInput(BaseModel):
    """日期时间工具输入参数"""
    operation: str = Field(description="操作类型: now, format, parse, add_days, diff")
    date_string: Optional[str] = Field(default=None, description="日期字符串")
    format_string: Optional[str] = Field(default=None, description="日期格式")
    days: Optional[int] = Field(default=None, description="要添加的天数")


class DateTimeTool(BaseTool):
    """日期时间工具"""
    name = "datetime_tool"
    description = "处理日期时间，支持格式化、解析、计算等操作"
    args_schema = DateTimeInput

    def _run(self, operation: str, date_string: Optional[str] = None, 
             format_string: Optional[str] = None, days: Optional[int] = None) -> str:
        try:
            if operation == "now":
                now = datetime.now()
                return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            
            elif operation == "format" and date_string and format_string:
                dt = datetime.strptime(date_string, format_string)
                return f"格式化后的时间: {dt.strftime('%Y年%m月%d日 %H时%M分%S秒')}"
            
            elif operation == "parse" and date_string:
                # 尝试解析常见格式
                formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%m/%d/%Y']
                for fmt in formats:
                    try:
                        dt = datetime.strptime(date_string, fmt)
                        return f"解析成功: {dt}"
                    except ValueError:
                        continue
                return "无法解析日期格式"
            
            elif operation == "add_days" and date_string and days:
                dt = datetime.strptime(date_string, '%Y-%m-%d')
                new_dt = dt + timedelta(days=days)
                return f"添加{days}天后的日期: {new_dt.strftime('%Y-%m-%d')}"
            
            elif operation == "diff" and date_string:
                # 计算与当前时间的差值
                dt = datetime.strptime(date_string, '%Y-%m-%d')
                now = datetime.now()
                diff = now - dt
                return f"与当前时间相差: {diff.days} 天"
            
            return f"操作 {operation} 执行完成"
        except Exception as e:
            return f"日期时间处理失败: {str(e)}"


# ==================== 压缩解压工具 ====================

class ZipToolInput(BaseModel):
    """压缩工具输入参数"""
    operation: str = Field(description="操作类型: compress, extract")
    source_path: str = Field(description="源文件或目录路径")
    target_path: str = Field(description="目标压缩包或解压目录路径")


class ZipTool(BaseTool):
    """压缩解压工具"""
    name = "zip_tool"
    description = "压缩文件或解压压缩包"
    args_schema = ZipToolInput

    def _run(self, operation: str, source_path: str, target_path: str) -> str:
        try:
            if operation == "compress":
                with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    if os.path.isfile(source_path):
                        zipf.write(source_path, os.path.basename(source_path))
                    else:
                        for root, dirs, files in os.walk(source_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, source_path)
                                zipf.write(file_path, arcname)
                return f"压缩完成: {target_path}"
            
            elif operation == "extract":
                with zipfile.ZipFile(source_path, 'r') as zipf:
                    zipf.extractall(target_path)
                return f"解压完成: {target_path}"
            
            return f"操作 {operation} 执行完成"
        except Exception as e:
            return f"压缩解压失败: {str(e)}"


# ==================== 工具管理器 ====================

class ToolManager:
    """工具管理器"""
    
    def __init__(self):
        self.tools = {}
        self._register_tools()
    
    def _register_tools(self):
        """注册所有工具"""
        # 文件操作工具
        self.tools["file_read"] = FileReadTool()
        self.tools["file_write"] = FileWriteTool()
        self.tools["file_list"] = FileListTool()
        
        # 网络请求工具
        self.tools["http_request"] = HttpRequestTool()
        
        # 数据处理工具
        self.tools["data_process"] = DataProcessTool()
        
        # 文本处理工具
        self.tools["text_process"] = TextProcessTool()
        
        # 数学计算工具
        self.tools["math_calc"] = MathCalcTool()
        
        # 日期时间工具
        self.tools["datetime_tool"] = DateTimeTool()
        
        # 压缩解压工具
        self.tools["zip_tool"] = ZipTool()
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取指定工具"""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        return list(self.tools.values())
    
    def list_tools(self) -> str:
        """列出所有可用工具"""
        tool_list = []
        for name, tool in self.tools.items():
            tool_list.append(f"- {name}: {tool.description}")
        return "可用工具:\n" + "\n".join(tool_list)
    
    def run_tool(self, tool_name: str, **kwargs) -> str:
        """运行指定工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            return f"工具 {tool_name} 不存在"
        
        try:
            return tool.run(kwargs)
        except Exception as e:
            return f"运行工具失败: {str(e)}"


# ==================== Ollama模型调用工具 ====================

class OllamaModelInput(BaseModel):
    """Ollama模型调用工具输入参数"""
    prompt: str = Field(description="要发送给模型的提示词")
    model_name: str = Field(default="llama2", description="模型名称")
    temperature: float = Field(default=0.7, description="温度参数")
    max_tokens: int = Field(default=1000, description="最大生成token数")


class OllamaModelTool(BaseTool):
    """Ollama模型调用工具"""
    name = "ollama_model"
    description = "调用本地Ollama大模型进行文本生成"
    args_schema = OllamaModelInput

    def _run(self, prompt: str, model_name: str = "llama2", 
             temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            # 初始化Ollama模型
            llm = Ollama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens
            )
            
            # 调用模型
            response = llm(prompt)
            return f"模型 {model_name} 响应:\n{response}"
        except Exception as e:
            return f"Ollama模型调用失败: {str(e)}"


# ==================== ReAct Agent ====================

class ReActAgent:
    """ReAct (Reasoning and Acting) Agent"""
    
    def __init__(self, model_name: str = "llama2", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            num_predict=2000
        )
        self.tool_manager = ToolManager()
        self.conversation_history = []
        
        # 创建ReAct提示模板
        self.prompt_template = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template="""你是一个智能助手，可以使用以下工具来帮助用户解决问题。

可用工具:
{tools}

工具名称: {tool_names}

使用以下格式:

Question: 需要回答的问题
Thought: 我需要思考如何解决这个问题
Action: 要使用的工具名称
Action Input: 工具的输入参数
Observation: 工具的输出结果
... (这个Thought/Action/Action Input/Observation可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

开始!

Question: {input}
Thought: {agent_scratchpad}"""
        )
        
        # 创建ReAct Agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tool_manager.get_all_tools(),
            prompt=self.prompt_template
        )
        
        # 创建Agent执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tool_manager.get_all_tools(),
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def chat(self, user_input: str) -> str:
        """与用户对话"""
        try:
            # 记录对话历史
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # 执行Agent
            response = self.agent_executor.invoke({"input": user_input})
            
            # 记录AI响应
            self.conversation_history.append({
                "role": "assistant", 
                "content": response["output"],
                "timestamp": datetime.now().isoformat()
            })
            
            return response["output"]
            
        except Exception as e:
            error_msg = f"处理请求时出错: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_history
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
    
    def save_conversation(self, file_path: str):
        """保存对话历史到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            return f"对话历史已保存到 {file_path}"
        except Exception as e:
            return f"保存对话历史失败: {str(e)}"
    
    def load_conversation(self, file_path: str):
        """从文件加载对话历史"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            return f"对话历史已从 {file_path} 加载"
        except Exception as e:
            return f"加载对话历史失败: {str(e)}"


# ==================== 增强的工具管理器 ====================

class EnhancedToolManager(ToolManager):
    """增强的工具管理器，包含Ollama模型调用"""
    
    def __init__(self):
        super().__init__()
        self._register_enhanced_tools()
    
    def _register_enhanced_tools(self):
        """注册增强工具"""
        # 添加Ollama模型工具
        self.tools["ollama_model"] = OllamaModelTool()
    
    def create_react_agent(self, model_name: str = "llama2", temperature: float = 0.7) -> ReActAgent:
        """创建ReAct Agent"""
        return ReActAgent(model_name=model_name, temperature=temperature)


# ==================== 使用示例 ====================

def main():
    """主函数 - 演示工具使用和ReAct Agent"""
    print("=== LangChain 工具集合 + ReAct Agent 演示 ===\n")
    
    # 创建增强工具管理器
    enhanced_manager = EnhancedToolManager()
    
    # 列出所有工具
    print("可用工具:")
    print(enhanced_manager.list_tools())
    print("\n" + "="*60 + "\n")
    
    # 示例1: 基础工具使用
    print("示例1: 基础工具使用")
    result = enhanced_manager.run_tool("file_write", 
                                     file_path="test.txt", 
                                     content="这是一个测试文件\n包含多行内容")
    print(result)
    
    result = enhanced_manager.run_tool("text_process", 
                                     text="我的邮箱是 test@example.com，电话是 123-456-7890", 
                                     operation="extract_email")
    print(result)
    print()
    
    # 示例2: ReAct Agent演示
    print("示例2: ReAct Agent 智能对话")
    print("注意: 需要确保Ollama服务正在运行，并且已安装llama2模型")
    print("安装命令: ollama pull llama2")
    print()
    
    try:
        # 创建ReAct Agent
        agent = enhanced_manager.create_react_agent(model_name="llama2", temperature=0.7)
        
        # 示例对话
        test_questions = [
            "请帮我创建一个名为'hello.txt'的文件，内容是'Hello World!'",
            "计算 15 * 8 + 32 的结果",
            "从文本'联系我：john@example.com 或 138-1234-5678'中提取邮箱和电话号码",
            "今天是2024年1月1日，30天后是什么日期？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"问题 {i}: {question}")
            print("-" * 50)
            
            try:
                response = agent.chat(question)
                print(f"回答: {response}")
            except Exception as e:
                print(f"处理问题时出错: {str(e)}")
                print("提示: 请确保Ollama服务正在运行")
            
            print("\n" + "="*60 + "\n")
        
        # 保存对话历史
        save_result = agent.save_conversation("conversation_history.json")
        print(f"对话历史保存结果: {save_result}")
        
    except Exception as e:
        print(f"创建ReAct Agent失败: {str(e)}")
        print("请确保已安装Ollama并运行服务")
        print("安装指南: https://ollama.ai/")
    
    # 清理测试文件
    try:
        os.remove("test.txt")
        print("测试文件已清理")
    except:
        pass


def interactive_chat():
    """交互式聊天模式"""
    print("=== ReAct Agent 交互式聊天 ===")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("输入 'save <filename>' 保存对话历史")
    print("输入 'load <filename>' 加载对话历史")
    print()
    
    try:
        enhanced_manager = EnhancedToolManager()
        agent = enhanced_manager.create_react_agent(model_name="llama2", temperature=0.7)
        
        while True:
            user_input = input("您: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("再见!")
                break
            
            elif user_input.lower() == 'clear':
                agent.clear_history()
                print("对话历史已清空")
                continue
            
            elif user_input.startswith('save '):
                filename = user_input[5:].strip()
                result = agent.save_conversation(filename)
                print(result)
                continue
            
            elif user_input.startswith('load '):
                filename = user_input[5:].strip()
                result = agent.load_conversation(filename)
                print(result)
                continue
            
            if not user_input:
                continue
            
            print("AI: ", end="")
            try:
                response = agent.chat(user_input)
                print(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
            
            print()
    
    except Exception as e:
        print(f"启动交互式聊天失败: {str(e)}")
        print("请确保已安装Ollama并运行服务")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        interactive_chat()
    else:
        main()