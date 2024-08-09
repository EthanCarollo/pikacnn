import tf from '@tensorflow/tfjs-node-gpu'

class BlackAndWhiteLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const input = inputs[0]; // tf.layers.Layer expects an array of inputs

      // Example: Convert to grayscale using the luminosity method
      const r = input.slice([0, 0, 0, 0], [-1, -1, -1, 1]);
      const g = input.slice([0, 0, 0, 1], [-1, -1, -1, 1]);
      const b = input.slice([0, 0, 0, 2], [-1, -1, -1, 1]);

      const grayscale = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));

      // Stack grayscale three times to create a pseudo RGB image
      const rgbImage = tf.concat([grayscale, grayscale, grayscale], 3);

      // Ensure output has shape (256, 256, 3)
      return rgbImage;
    });
  }

  static get className() {
    return 'BlackAndWhiteLayer';
  }
}

tf.serialization.registerClass(BlackAndWhiteLayer);

export function getBlackAndWhiteLayer(){
  return new BlackAndWhiteLayer()
}

export function getBlackAndWhiteLayerInput(imgSize){
  return new BlackAndWhiteLayer({inputShape: [imgSize, imgSize, 3]})
}