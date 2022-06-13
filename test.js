const tf = require('@tensorflow/tfjs-node-gpu');
(async function(){
    const Model = require('./Model');
    let model = new Model({
        encoder:'vocab'
    });
    model.loadTextFile('./data/shakespeare.txt');
  //  model.model.summary();
    await model.train();
    /*
    await model.sequencializer.sequences.take(1).forEachAsync(function(s){
        console.log(s);
    });*/
  //  let xBuffer = tf.buffer([64,100,66]);
  //  console.log(xBuffer.toTensor());
})();