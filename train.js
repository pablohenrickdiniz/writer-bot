const TokenNetwork = require('./TokenNetwork');
const fs = require('fs');
const path = require('path');
const TextUtils = require('./TextUtils');
const dataDir = './data';


async function init(){
    let network = new TokenNetwork();
    await network.load('./models/resumo');
    network.loadTextFile('./data/resumo.txt');
    while(true){
        fs.writeFileSync('resumo.txt',network.predict('A terra era sem forma e va'));
        await network.train(100,function(logs,epochs,loss){
            let log = logs.toString().padStart(epochs.toString().length,'0')+'/'+epochs+' - treinando (loss:'+loss+')';
            console.log(log);
        });
       
        network.save('./models/resumo');
    }
    //fs.writeFileSync('training-data.json',JSON.stringify(network.trainingData,null,4));
    /*
    while(true){
        const files = fs.readdirSync(dataDir).map((f) => path.join(dataDir,f));
        for(let i = 0; i < files.length;i++){
            let file = files[i];
            let name = path.basename(file).split('.')[0];
            let book = new BookNetwork(name,modelsDir);
            let contents = fs.readFileSync(file,{encoding:'utf-8'});
            contents = BookNetwork.cleanText(contents);
            contents = contents.split("\n");
            
            for(let l = 0; l < contents.length;l++){
                book.add(l,contents[l]);
            }
         
            let outputfile  = path.join(outputDir,book.getID()+'-'+name+'.txt');
            await book.train(100);
            for(let i = 0; i < 10;i++){
                console.log('line '+(i+1)+': '+(await book.predict(i)));
            }
            let text = (await book.predict(0));
            fs.appendFileSync(outputfile,text+"\n");
        }
    }
    */
}
init();