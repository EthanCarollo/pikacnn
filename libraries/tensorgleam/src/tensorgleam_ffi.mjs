import tf from '@tensorflow/tfjs'

export function logModel(model){
    console.log(model)
}

export function getSequentialModel(){
    return tf.sequential();
}