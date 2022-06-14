function split(text,sep){
    if(sep.constructor !== [].constructor){
        sep = [sep];
    }
    let pieces = [];
    for(let i = 0; i < text.length;i++){
        let chr = text.charAt(i);
        if(sep.indexOf(chr) !== -1){
            let piece = text.substring(0,i);
            pieces.push(piece);
            pieces.push(chr);
            text = text.substring(i+1);
            i = 0;
        }
    }
    if(text.length > 0){
        pieces.push(text);
    }
    return pieces;
}

let TextUtils = {
    split:split
};

module.exports = TextUtils;