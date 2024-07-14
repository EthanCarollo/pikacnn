import tf from '@tensorflow/tfjs-node'

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

export async function modelFit(model, trainData, validationSplit, epochs = 30, stepsPerEpoch = 100, validationSteps = 50){
    const history = await model.fit(trainData, {
        epochs: epochs,
        validationSplit: validationSplit,
        stepsPerEpoch: stepsPerEpoch,
        validationSteps: validationSteps
      });
    return history;
}

export async function modelSave(model, path){
    return await model.save(path)
}

/**
 * 
 */

export function decodeImage(buffer){
    return tf.node.decodeImage(buffer)
}

export function tensorResizeNearestNeighbor(tensor, tensor_size){
    return tensor.resizeNearestNeighbor([tensor_size, tensor_size])
}

export function tensorToFloat(tensor){
    return tensor.toFloat()
}

export function tensorDivScalar(tensor, scalarFactor){
    return tensor.div(tf.scalar(scalarFactor))
}

export function tensorExpandDims(tensor){
    return tensor.expandDims()
}