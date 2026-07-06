<script setup>
const props = defineProps({
  graphType: String,
})

const emit = defineEmits(['change-graph'])

const graphs = [
  { id: 'graph1', name: '基础 RAG', desc: 'Agent + 检索评估', icon: '🔄' },
  { id: 'graph2', name: 'Corrective RAG', desc: '路由 + 幻觉检测', icon: '🎯' },
]
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <h1 class="logo">RAG Chat</h1>
    </div>

    <div class="graph-selector">
      <div class="section-label">选择模式</div>
      <button
        v-for="g in graphs"
        :key="g.id"
        :class="['graph-btn', { active: graphType === g.id }]"
        @click="emit('change-graph', g.id)"
      >
        <span class="graph-icon">{{ g.icon }}</span>
        <div class="graph-info">
          <div class="graph-name">{{ g.name }}</div>
          <div class="graph-desc">{{ g.desc }}</div>
        </div>
      </button>
    </div>

    <div class="sidebar-footer">
      <div class="status">
        <span class="dot"></span>
        当前: {{ graphs.find(g => g.id === graphType)?.name }}
      </div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 240px;
  min-width: 240px;
  background: var(--sidebar-bg);
  color: var(--sidebar-text);
  display: flex;
  flex-direction: column;
  border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.sidebar-header {
  padding: 20px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.logo {
  font-size: 18px;
  font-weight: 700;
  color: #fff;
  margin: 0;
  letter-spacing: -0.5px;
}

.graph-selector {
  padding: 16px 12px;
  flex: 1;
}

.section-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #7a7c85;
  margin-bottom: 8px;
  padding-left: 4px;
}

.graph-btn {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--sidebar-text);
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
  margin-bottom: 4px;
}

.graph-btn:hover {
  background: rgba(255, 255, 255, 0.06);
}

.graph-btn.active {
  background: var(--sidebar-active);
  color: #fff;
}

.graph-icon {
  font-size: 18px;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 6px;
}

.graph-btn.active .graph-icon {
  background: rgba(255, 255, 255, 0.2);
}

.graph-name {
  font-size: 13px;
  font-weight: 600;
}

.graph-desc {
  font-size: 11px;
  opacity: 0.7;
  margin-top: 1px;
}

.sidebar-footer {
  padding: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  opacity: 0.7;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
}
</style>
