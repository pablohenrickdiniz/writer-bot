const fs = require('fs');
const modelDir = '/content/drive/MyDrive/ia-projects/writer-bot/models/biblia';
const outputFile = '/content/drive/MyDrive/ia-projects/writer-bot/output/biblia.txt';
const perf = require('execution-time')();
(async function(){
    const Model = require('./Model');
    let model = new Model({
        encoder:'vocab'
    });
    let loaded = false;
    try{
        perf.start('loading model');
        loaded = await model.load(modelDir);
        console.log('loading model '+perf.stop('loading model').preciseWords);
    }
    catch(ex){
        
    }

    if(!loaded){
        model.loadTextFile('./data/biblia.txt');
    }

    let result = await model.generate(10);
    fs.writeFileSync(outputFile,result);
    let epochs = 100;
    await model.train(epochs,async function(index,loss){
        await model.save(modelDir);
        let result = await model.generate(10);
        fs.writeFileSync(outputFile,result);
    });
})();