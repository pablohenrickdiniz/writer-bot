"use strict";

var TokenNetwork = require('./TokenNetwork');

var fs = require('fs');

var path = require('path');

var TextUtils = require('./TextUtils');

var dataDir = './data';

function init() {
  var contents, network;
  return regeneratorRuntime.async(function init$(_context) {
    while (1) {
      switch (_context.prev = _context.next) {
        case 0:
          contents = fs.readFileSync("./data/biblia.txt", {
            encoding: 'utf-8'
          });
          network = new TokenNetwork();
          network.loadText(contents);
          fs.writeFileSync('training-data.json', JSON.stringify(network.trainingData, null, 4));
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

        case 4:
        case "end":
          return _context.stop();
      }
    }
  });
}

init();