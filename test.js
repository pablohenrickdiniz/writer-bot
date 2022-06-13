const Vocab = require('./Vocab');
const Tokenizer = require('./Tokenizer');


let vocab = new Vocab();
let tokenizer = new Tokenizer();

tokenizer.loadText("Esse é um texto de teste");
vocab.loadText("Esse é um texto de teste");
console.log(tokenizer.encode("Esse é um texto de teste"));
console.log(vocab.encode("Esse é um texto de teste"));