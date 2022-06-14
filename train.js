const fs = require('fs');
const modelDir = '/content/drive/MyDrive/ia-projects/writer-bot/models/biblia';
const outputFile = '/content/drive/MyDrive/ia-projects/writer-bot/output/biblia.txt';
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
        model.loadTextFile('./data/biblia.txt');
    }
    
    let epochs = 100;
    await model.train(epochs,async function(index,loss){
        await model.save(modelDir);
        let result = await model.generate(1024);
        fs.writeFileSync(outputFile,result);
    });
})();