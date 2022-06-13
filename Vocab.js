function Vocab(text){
    let self = this;
    initialize(self);
    if(text){
        self.loadText(text);
    }
}

function initialize(self){
    let items = [];

    let encode = function(text){
        return text.split('').map((c) => items.indexOf(c));
    };

    let decode =  function(indexes){
        return indexes.map((i) => items[i]?items[i]:" ").join('');
    };

    let loadText = function(text){
        items = [...new Set(items.concat(text.split('')))].sort();
    };
    
    Object.defineProperty(self,'items',{
        get:function(){
            return [...items];
        }
    });

    Object.defineProperty(self,'size',{
        get:function(){
            return items.length;
        }
    });

    Object.defineProperty(self,'encode',{
        get:function(){ return encode;}
    });

    Object.defineProperty(self,'decode',{
        get:function(){ return decode;}
    });

    Object.defineProperty(self,'loadText',{
        get:function(){ return loadText;}
    });
}

module.exports = Vocab;