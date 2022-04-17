const BookNetwork = require('./BookNetwork');
const fs = require('fs');
const path = require('path');
//const dataDir = './data';
//const outputDir = './output';
//const modelsDir = './models';
const dataDir = '/content/drive/MyDrive/ia-projects/writer-bot/data';
const outputDir = '/content/drive/MyDrive/ia-projects/writer-bot/output';
const modelsDir = '/content/drive/MyDrive/ia-projects/writer-bot/models';
if(!fs.existsSync(outputDir)){
    fs.mkdirSync(outputDir,{
        recursive:true
    });
}

async function init(){
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
                book.add(l,contents[i]);
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
}
init();