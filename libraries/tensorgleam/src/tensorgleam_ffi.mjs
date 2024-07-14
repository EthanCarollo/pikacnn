import tf from '@tensorflow/tfjs'

export function logModel(model){
    console.log(model)
}

export function getSequentialModel(){
    return tf.sequential();
}

export function addLayerToModel(model, layer){
    model.add(layer)
    return model
}

/**
 * @param {Array} inputShape 
 * @param {Int16Array} filters 
 * @param {Int16Array} kernelSize 
 * @param {string} activation 
 * @returns 
 */
export function getConvolution2DLayer(inputShape, filters, kernelSize, activation){
    return tf.layers.conv2d({
        inputShape: inputShape,
        filters: filters,
        kernelSize: kernelSize,
        activation: activation
    })
}

/**
 * @param {Int16Array} filters 
 * @param {Int16Array} kernelSize 
 * @param {string} activation 
 * @returns 
 */
export function getConvolution2DLayerNoInput(filters, kernelSize, activation){
    return tf.layers.conv2d({
        filters: filters,
        kernelSize: kernelSize,
        activation: activation
    })
}

/**
 * @param {Array} poolSize array like [x, x] 
 * @returns 
 */
export function getMaxPooling2D(poolSize){
    return tf.layers.maxPooling2d({poolSize: poolSize})
}

export function getFlatten(){
    return tf.layers.flatten()
}

/**
 * @param {Float32Array} rate 
 * @returns 
 */
export function getDropOut(rate){
    return tf.layers.dropout({rate: rate})
}

/**
 * @param {Int16Array} units 
 * @param {String} activation 
 * @returns 
 */
export function getDense(units, activation){
    return tf.layers.dense({units: units, activation: activation})
}

export function modelCompile(model, optimizer, loss, metrics){
    model.compile({
        optimizer: optimizer,
        loss: loss,
        metrics: metrics
    });
    return model
}

export function modelSummary(model){
    model.summary()
    return model
}