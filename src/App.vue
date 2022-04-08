<script setup lang="ts">
import 'bootstrap/dist/css/bootstrap.min.css'
import 'bootstrap/dist/js/bootstrap.min.js'
import * as tf from '@tensorflow/tfjs';
import { onMounted, ref } from 'vue';

let drawCanvas: HTMLCanvasElement | null
let dataCanvas: HTMLCanvasElement | null
let drawCanvasCtx: CanvasRenderingContext2D | null | undefined
let dataCanvasCtx: CanvasRenderingContext2D | null | undefined
let targetNum = ref(0)
let trainStatus = ref("")
let result = ref("")

let trainingModel: tf.Sequential

onMounted(() => {
  drawCanvas = document.querySelector('#drawCanvas')
  dataCanvas = document.querySelector('#dataCanvas')
  drawCanvasCtx = drawCanvas?.getContext('2d')
  dataCanvasCtx = dataCanvas?.getContext('2d')
  if (drawCanvasCtx) {
    drawCanvasCtx.lineWidth = 16
    drawCanvasCtx.lineCap = "round"
    drawCanvasCtx.lineJoin = "round"
  }
  loadOrCreateModel()
})
let drawing: boolean
const canvasMouseDownHandler = (e: MouseEvent) => {
  drawing = true
  drawCanvasCtx?.beginPath()
  drawCanvasCtx?.moveTo(e.offsetX, e.offsetY)
}

const canvasMouseMoveHandler = (e: MouseEvent) => {
  if (drawing) {
    drawCanvasCtx?.lineTo(e.offsetX, e.offsetY)
    drawCanvasCtx?.stroke()
  }
}

const canvasMouseUpHandler = () => {
  drawing = false
  if (dataCanvasCtx) {
    dataCanvasCtx.fillStyle = "white"
    dataCanvasCtx.fillRect(0, 0, 50, 30)
    dataCanvasCtx.drawImage(
      //@ts-ignore
      drawCanvas,
      0, 0, 50, 30
    )
  }
}

//加载本地模型或者创建一个新的模型
const loadOrCreateModel = async () => {
  try {
    trainingModel = await tf.loadLayersModel('localstorage://my-model')
  } catch (e) {
    console.warn("Can not load trainingModel from LocalStorage, create a new trainingModel")
    trainingModel = tf.sequential({
      layers: [//层数
        tf.layers.inputLayer({ inputShape: [1500] }), //输入的张量
        tf.layers.dense({ units: 100 }), //输出空间的大小
        tf.layers.softmax() //把结果按照概率分配到输出空间上，输出空间中所有值加在一起为一
      ]
    })


  }
  // 编译模型
  trainingModel.compile({
    optimizer: 'sgd',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })
}

const btnClearCanvasClickedHandler = () => {
  drawCanvasCtx?.clearRect(0, 0, drawCanvas?.width, drawCanvas?.height)
  dataCanvasCtx?.clearRect(0, 0, dataCanvas?.width, drawCanvas?.height)
}

const getImageData = () => {
  let imageData = dataCanvasCtx?.getImageData(0, 0, 50, 30) //6000 20 × 50 × 4 因为每个像素点由四个字节组成分别存储RGBA
  let pixelData = [] //只存储黑白数据

  let color
  for (let i = 0; i < imageData?.data.length; i += 4) {
    color = (imageData?.data[i] + imageData?.data[i + 1] + imageData?.data[i + 2]) / 3//RGB为一组
    pixelData.push(Math.round((255 - color) / 255));//取反值，让特征变得明显,让数据简化，只有0/1
  }
  return pixelData
}

const btnTrainClickedHandler = async () => {
  console.log(targetNum)
  let data = getImageData()
  //[1,0,0,0,0,0,0,0,0,0] 0
  //[0,1,0,0,0,0,0,0,0,0] 1
  let targetTensor = tf.oneHot(Number(targetNum.value), 100) //返回一个一阶张量，代表targetNum
  console.log("Start training")
  await trainingModel.fit(tf.tensor([data]), tf.tensor([targetTensor.arraySync()]), {//用 fit 函数来训练
    epochs: 100,
    callbacks: {
      onEpochEnd(epoch: number, logs: tf.Logs | undefined) { //在某一个阶段完成的时候
        trainStatus.value = `<div>Step: ${epoch}</div><div>Loss: ${logs?.loss}</div>`;
      }
    }
  })
  trainStatus.value = `<div style="color: green">训练完成</div>`
  console.log("Completed")

  //把训练模型保存在本地
  await trainingModel.save('localstorage://my-model')
}

const btnPredictClickedHandler = async () => {
  let data = getImageData()
  let predictions = await trainingModel.predict(tf.tensor([data]))
  result.value = predictions.argMax(1).arraySync()[0] //argMax 默认情况下获得一个一维数组最大值所在的位置
}


</script>

<template>
  <div class="cardContainer">
    <div class="card">
      <div class="card-header">在此添加数字</div>
      <div class="card-body">
        <canvas
          width="500"
          height="300"
          id="drawCanvas"
          @mousedown="canvasMouseDownHandler"
          @mousemove="canvasMouseMoveHandler"
          @mouseup="canvasMouseUpHandler"
        ></canvas>
        <button class="btn btn-primary" @click="btnClearCanvasClickedHandler">清空</button>
      </div>
      <div class="card-header">图像数据预览</div>
      <div class="card-body">
        <canvas width="50" height="30" style="border-style:solid" id="dataCanvas"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-header">训练</div>
      <div class="card-body">
        关联数字:
        <input type="text" v-model="targetNum" />
        <button
          class="btn btn-primary"
          @click="btnTrainClickedHandler"
          style="margin-left:8px;margin-top:-2px"
        >开始训练</button>
        <div>
          <div v-html="trainStatus"></div>
        </div>
      </div>
      <div class="card-header">识别结果</div>
      <div class="card-body">
        <button class="btn btn-primary" @click="btnPredictClickedHandler">预测</button>
        <div>{{ result }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.card {
  margin: 10px;
}
.cardContainer {
  display: flex;
  padding: 10px;
}
.card-body {
  justify-self: center;
  text-align: center;
}
#canvas {
  border-style: dashed;
  display: block;
}
button {
  margin-top: 10px;
}
</style>
