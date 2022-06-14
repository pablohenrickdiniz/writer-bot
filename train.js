const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
(async function(){
    const Model = require('./Model');
    let model = new Model({
        encoder:'vocab'
    });
    model.loadTextFile('./data/biblia.txt');
    let epochs = 100;
    await model.train(epochs,async function(index,loss){
        console.log(index+'/'+epochs+' - treinando, taxa de erro:'+loss.toFixed(8));
        let res = fs.createWriteStream('biblia-gerada.txt','utf-8');
        await model.generate(1000,function(t){
            res.write(t);
        });
        res.close();
    });
    /*
    await model.sequencializer.sequences.take(1).forEachAsync(function(s){
        console.log(s);
    });*/
  //  let xBuffer = tf.buffer([64,100,66]);
  //  console.log(xBuffer.toTensor());
})();