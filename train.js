const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const modelDir = '/content/drive/MyDrive/ia-projects/writer-bot/models/shakespeare';
const outputFile = '/content/drive/MyDrive/ia-projects/writer-bot/output/shakespeare.txt';
(async function(){
    const Model = require('./Model');
    let model = new Model({
        encoder:'vocab'
    });
    let loaded = false;
    try{
        loaded = await model.load(modelDir);
    }
    catch(ex){
        
    }

    if(!loaded){
        model.loadTextFile('./data/shakespeare.txt');
    }
    
    let epochs = 100;
    await model.train(epochs,async function(index,loss){
        await model.save(modelDir);
        console.log(index+'/'+epochs+' - treinando, taxa de erro:'+loss.toFixed(8));
    });
})();