import fs from 'fs';
import PklParser from './pklparser';

export function pklDecode(path) {
    let config = fs.readFileSync(path, 'utf8')
    let pklParser = new PklParser(config)
    return pklParser.parse()
}
