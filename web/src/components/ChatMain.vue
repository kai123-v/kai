<script setup>
import { ref, nextTick, watch } from 'vue'
import { marked } from 'marked'
import NodeTimeline from './NodeTimeline.vue'

const props = defineProps({
  graphType: String,
  threadId: String,
})

const emit = defineEmits(['session-created'])

const messages = ref([])
const inputText = ref('')
const loading = ref(false)
const currentNode = ref('')
const nodes = ref([])
const chatContainer = ref(null)

// 监听 graphType 变化，清空对话
watch(() => props.graphType, () => {
  messages.value = []
  nodes.value = []
  currentNode.value = ''
})

function scrollToBottom() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

async function sendMessage() {
  const question = inputText.value.trim()
  if (!question || loading.value) return

  inputText.value = ''
  loading.value = true
  nodes.value = []
  currentNode.value = '正在处理...'

  // 添加用户消息
  messages.value.push({ role: 'user', content: question })
  scrollToBottom()

  // 添加 AI 消息占位
  const aiIdx = messages.value.length
  messages.value.push({ role: 'assistant', content: '' })
  scrollToBottom()

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        graph_type: props.graphType,
        thread_id: props.threadId,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // 解析 SSE 数据（以 data: 开头的行）
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // 保留未完成的行

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const dataStr = line.slice(5).trim()
          if (!dataStr) continue
          try {
            const data = JSON.parse(dataStr)

            if (data.type === 'session') {
              emit('session-created', data.thread_id)
            } else if (data.type === 'node') {
              currentNode.value = data.label || data.node
              nodes.value.push({
                node: data.node,
                label: data.label,
                detail: data.detail || '',  // 后端已拼好 detail 字符串
                timestamp: Date.now(),
              })
              scrollToBottom()
            } else if (data.type === 'answer') {
              messages.value[aiIdx].content = data.generation
              currentNode.value = '回答完成'
              scrollToBottom()
            } else if (data.type === 'error') {
              messages.value[aiIdx].content = `出错了: ${data.message}`
              currentNode.value = '错误'
            } else if (data.type === 'done') {
              loading.value = false
              currentNode.value = ''
            }
          } catch (e) {
            // 忽略非 JSON 行
          }
        }
      }
    }
  } catch (e) {
    messages.value[aiIdx].content = `请求失败: ${e.message}`
    currentNode.value = '错误'
  } finally {
    loading.value = false
  }
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text)
}
</script>

<template>
  <main class="chat-main">
    <!-- 顶部状态栏 -->
    <header class="chat-header">
      <div class="header-left">
        <span class="header-title">RAG Chat</span>
        <span class="header-graph">{{ graphType === 'graph1' ? '基础 RAG' : 'Corrective RAG' }}</span>
      </div>
      <div v-if="currentNode" class="header-status">
        <span class="spinner"></span>
        <span>{{ currentNode }}</span>
      </div>
    </header>

    <!-- 消息区域 -->
    <div class="chat-messages" ref="chatContainer">
      <!-- 空状态 -->
      <div v-if="messages.length === 0" class="empty-state">
        <div class="empty-icon">💬</div>
        <div class="empty-title">开始对话</div>
        <div class="empty-desc">输入问题，AI 将基于知识库检索并回答</div>
      </div>

      <!-- 消息列表 -->
      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        :class="['message', msg.role]"
      >
        <div class="msg-avatar">
          {{ msg.role === 'user' ? '👤' : '🤖' }}
        </div>
        <div class="msg-content">
          <div
            v-if="msg.role === 'user'"
            class="msg-text"
          >{{ msg.content }}</div>
          <div
            v-else
            class="msg-text markdown-body"
            v-html="renderMarkdown(msg.content || (loading && idx === messages.length - 1 ? '思考中...' : ''))"
          ></div>
        </div>
      </div>
    </div>

    <!-- 节点时间线 -->
    <NodeTimeline v-if="nodes.length > 0" :nodes="nodes" />

    <!-- 输入区域 -->
    <div class="chat-input-area">
      <div class="input-wrapper">
        <textarea
          v-model="inputText"
          @keydown="handleKeydown"
          placeholder="输入问题... (Enter 发送)"
          rows="1"
          :disabled="loading"
        ></textarea>
        <button
          class="send-btn"
          @click="sendMessage"
          :disabled="loading || !inputText.trim()"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
          </svg>
        </button>
      </div>
    </div>
  </main>
</template>

<style scoped>
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  min-width: 0;
}

.chat-header {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: var(--chat-bg);
  border-bottom: 1px solid var(--border);
}

.header-title {
  font-weight: 700;
  font-size: 15px;
  color: var(--text);
}

.header-graph {
  margin-left: 10px;
  font-size: 12px;
  padding: 2px 10px;
  border-radius: 12px;
  background: var(--accent-light);
  color: var(--accent);
  font-weight: 500;
}

.header-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--accent);
}

.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid var(--accent-light);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  opacity: 0.5;
}

.empty-icon {
  font-size: 48px;
}

.empty-title {
  font-size: 18px;
  font-weight: 600;
}

.empty-desc {
  font-size: 14px;
  color: var(--text-secondary);
}

.message {
  display: flex;
  gap: 10px;
  max-width: 80%;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.message.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.message.assistant {
  align-self: flex-start;
}

.msg-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  background: var(--node-bg);
  flex-shrink: 0;
}

.message.user .msg-avatar {
  background: var(--accent);
}

.msg-content {
  padding: 10px 14px;
  border-radius: var(--radius);
  font-size: 14px;
  line-height: 1.7;
  max-width: 100%;
  word-break: break-word;
}

.message.user .msg-content {
  background: var(--user-msg);
  color: var(--user-msg-text);
  border-bottom-right-radius: 4px;
}

.message.assistant .msg-content {
  background: var(--ai-msg);
  color: var(--ai-msg-text);
  border-bottom-left-radius: 4px;
}

.chat-input-area {
  padding: 16px 24px 20px;
  background: var(--chat-bg);
  border-top: 1px solid var(--border);
}

.input-wrapper {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px 12px;
  transition: border-color 0.2s;
}

.input-wrapper:focus-within {
  border-color: var(--accent);
}

textarea {
  flex: 1;
  border: none;
  outline: none;
  resize: none;
  font-size: 14px;
  line-height: 1.5;
  background: transparent;
  color: var(--text);
  font-family: inherit;
  max-height: 120px;
}

textarea::placeholder {
  color: var(--text-secondary);
}

.send-btn {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 8px;
  background: var(--accent);
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.send-btn:hover:not(:disabled) {
  background: #5b4bd5;
}

.send-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>
