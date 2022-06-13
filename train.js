const tf = require('@tensorflow/tfjs-node-gpu');
(async function(){
    const Model = require('./Model');
    let model = new Model({
        encoder:'vocab'
    });
    model.loadTextFile('./data/biblia.txt');
  //  model.model.summary();
    let epochs = 100;
    await model.train(epochs,async function(index,loss){
        console.log(index+'/'+epochs+' - treinando, taxa de erro:'+loss.toFixed(8));
        let text = await model.generate(10000);
        fs.writeFileSync('biblia-gerada.txt',text);
    });

    /*
    await model.sequencializer.sequences.take(1).forEachAsync(function(s){
        console.log(s);
    });*/
  //  let xBuffer = tf.buffer([64,100,66]);
  //  console.log(xBuffer.toTensor());
})();