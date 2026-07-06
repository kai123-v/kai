<script setup>
import { ref, nextTick } from 'vue'
import ChatSidebar from './components/ChatSidebar.vue'
import ChatMain from './components/ChatMain.vue'

const graphType = ref('graph2')
const threadId = ref(null)

function onGraphChange(type) {
  graphType.value = type
  threadId.value = null // 切换 graph 时重置会话
}
</script>

<template>
  <ChatSidebar :graphType="graphType" @change-graph="onGraphChange" />
  <ChatMain :graphType="graphType" :threadId="threadId" @session-created="(id) => threadId = id" />
</template>

<style scoped>
</style>
