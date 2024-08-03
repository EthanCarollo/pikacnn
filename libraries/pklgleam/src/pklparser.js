export default class PklParser {
    constructor(data) {
        this.lines = data.split('\n').map(line => line.trim()).filter(line => line.length > 0);
        this.currentLine = 0;
    }

    parse() {
        let result = {};
        while (this.currentLine < this.lines.length) {
            const line = this.lines[this.currentLine];
            if (line.includes('=') && !line.includes('{')) {
                const [key, value] = line.split('=').map(part => part.trim());
                result[key] = this.parseValue(value);
                this.currentLine++;
            } else if (line.includes('{')) {
                const [key] = line.split('{').map(part => part.trim());
                this.currentLine++;
                result[key] = this.parseBlock();
            }
        }
        return result;
    }

    parseValue(value) {
        if (value.startsWith('"') && value.endsWith('"')) {
            return value.slice(1, -1);
        } else if (!isNaN(value)) {
            return Number(value);
        } else {
            return value;
        }
    }

    parseBlock() {
        let block = {};
        while (this.currentLine < this.lines.length && !this.lines[this.currentLine].includes('}')) {
            const line = this.lines[this.currentLine];
            if (line.includes('=')) {
                const [key, value] = line.split('=').map(part => part.trim());
                block[key] = this.parseValue(value);
            }
            this.currentLine++;
        }
        this.currentLine++;
        return block;
    }
}
