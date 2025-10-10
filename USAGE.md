# LangChain 工具集合 + ReAct Agent

这是一个基于LangChain的完整工具集合，集成了本地Ollama大模型和ReAct（Reasoning and Acting）框架。

## 🚀 功能特性

### 基础工具
- **文件操作**: 读取、写入、列出文件
- **网络请求**: HTTP请求工具
- **数据处理**: JSON数据处理
- **文本处理**: 文本清理、提取、统计
- **数学计算**: 表达式计算
- **日期时间**: 日期格式化、计算
- **压缩解压**: 文件压缩和解压

### AI功能
- **Ollama模型调用**: 本地大模型推理
- **ReAct Agent**: 智能推理和行动框架
- **对话历史管理**: 保存和加载对话记录
- **交互式聊天**: 实时对话模式

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🔧 Ollama设置

1. 安装Ollama: https://ollama.ai/
2. 下载模型:
```bash
ollama pull llama2
# 或其他模型如:
ollama pull codellama
ollama pull mistral
```

3. 启动Ollama服务:
```bash
ollama serve
```

## 💻 使用方法

### 1. 基础演示
```bash
python tool.py
```

### 2. 交互式聊天
```bash
python tool.py chat
```

### 3. 编程使用
```python
from tool import EnhancedToolManager, ReActAgent

# 创建工具管理器
manager = EnhancedToolManager()

# 创建ReAct Agent
agent = manager.create_react_agent(model_name="llama2")

# 开始对话
response = agent.chat("请帮我计算 15 * 8 + 32 的结果")
print(response)

# 保存对话历史
agent.save_conversation("my_chat.json")
```

## 🎯 ReAct框架说明

ReAct（Reasoning and Acting）是一个结合推理和行动的框架：

1. **Thought**: AI思考如何解决问题
2. **Action**: 选择并使用合适的工具
3. **Observation**: 观察工具执行结果
4. **Final Answer**: 基于观察给出最终答案

## 📝 示例对话

```
您: 请帮我创建一个名为'hello.txt'的文件，内容是'Hello World!'

AI: 
Thought: 用户想要创建一个文件，我需要使用file_write工具来完成这个任务。
Action: file_write
Action Input: {"file_path": "hello.txt", "content": "Hello World!"}
Observation: 文件 hello.txt 写入成功，内容长度: 12 字符
Thought: 文件创建成功，任务完成。
Final Answer: 我已经成功创建了名为'hello.txt'的文件，内容为'Hello World!'。
```

## 🛠️ 自定义工具

您可以轻松添加自定义工具：

```python
class CustomTool(BaseTool):
    name = "custom_tool"
    description = "自定义工具描述"
    
    def _run(self, input_param: str) -> str:
        # 实现工具逻辑
        return "工具执行结果"

# 注册到管理器
manager.tools["custom_tool"] = CustomTool()
```

## 🔍 故障排除

1. **Ollama连接失败**: 确保Ollama服务正在运行
2. **模型不存在**: 使用 `ollama list` 检查已安装的模型
3. **依赖安装失败**: 尝试使用虚拟环境

## 📄 许可证

MIT License
