
function cleanWord(word){
    return [...word.trim()].filter((chr) => chr.length > 0).join('');
}

function cleanLine(text){
    return text.trim().split(/\s+/g).map(function(word){
        return cleanWord(word);
    }).filter(function(word){
        return word.length > 0;
    }).join(' ');
}

function cleanText(text){
    return text.trim().split(/[\r\n]+/).map(function(line){
        return cleanLine(line);
    }).filter((l) => l.length > 0).join("\n");
}

function split(text,sep){
    if(sep.constructor !== [].constructor){
        sep = [sep];
    }
    let pieces = [];
    let i = 0;
    let j = 0;
    for(i = 0, j = 0; i < text.length;i++){
        let chr = text.charAt(i);
        if(sep.indexOf(chr) !== -1){
            if(j < i){
                pieces.push(text.slice(j,i));
            }
            pieces.push(chr);
            j = i+1;
        }
    }
    if(j < i){
        pieces.push(text.slice(j,i));
    }
    return pieces;
}

let TextUtils = {
    split:split,
    cleanText:cleanText
};

module.exports = TextUtils;