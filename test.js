const BookNetwork = require('./BookNetwork');
const fs = require('fs');
const path = require('path');
const dataDir = './data';
const outputDir = './output';
const modelsDir = './models';
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
            let text = (await book.predict(0));
            fs.appendFileSync(outputfile,text+"\n");
        }
    }
}
init();