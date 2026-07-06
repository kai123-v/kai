<script setup>
const props = defineProps({
  nodes: Array,
})

// 节点图标映射
const nodeIcons = {
  user_input: '👤',
  agent: '🤖',
  retrieve: '📚',
  rewrite: '✏️',
  generate: '✨',
  grade_documents: '🔍',
  web_search: '🌐',
  transformer_query: '🔄',
  route_question: '🧭',
}
</script>

<template>
  <div class="node-timeline">
    <div class="timeline-title">检索过程</div>
    <div class="timeline-items">
      <div
        v-for="(node, idx) in nodes"
        :key="idx"
        class="timeline-item"
        :class="{ latest: idx === nodes.length - 1 }"
      >
        <div class="timeline-dot"></div>
        <div v-if="idx < nodes.length - 1" class="timeline-line"></div>
        <div class="timeline-content">
          <span class="timeline-icon">{{ nodeIcons[node.node] || '⚙️' }}</span>
          <span class="timeline-label">{{ node.label }}</span>
          <span v-if="node.detail" class="timeline-detail">{{ node.detail }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.node-timeline {
  border-top: 1px solid var(--border);
  background: var(--chat-bg);
  padding: 12px 24px;
  max-height: 160px;
  overflow-y: auto;
}

.timeline-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.timeline-items {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.timeline-item {
  position: relative;
  display: inline-flex;
  align-items: center;
}

.timeline-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  flex-shrink: 0;
}

.timeline-item.latest .timeline-dot {
  background: var(--success);
  box-shadow: 0 0 0 3px rgba(0, 184, 148, 0.2);
}

.timeline-line {
  width: 16px;
  height: 1px;
  background: var(--border);
  margin: 0 2px;
}

.timeline-content {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 16px;
  background: var(--node-bg);
  font-size: 12px;
  white-space: nowrap;
}

.timeline-item.latest .timeline-content {
  background: var(--node-active-bg);
  color: var(--node-active);
  font-weight: 500;
}

.timeline-icon {
  font-size: 12px;
}

.timeline-label {
  font-weight: 500;
}

.timeline-detail {
  color: var(--text-secondary);
  font-size: 11px;
}

.timeline-item.latest .timeline-detail {
  color: var(--accent);
  opacity: 0.8;
}
</style>
