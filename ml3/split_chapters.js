const fs = require('fs');
const path = require('path');

const inputFile = '/mnt/g/deeplearn/ml3/backup/ml3.md';
const outputDir = '/mnt/g/deeplearn/ml3/chapter';

fs.readFile(inputFile, 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading the file:', err);
        return;
    }

    const chapters = data.split(/\n## /);
    let chapterCount = 0;

    chapters.forEach((chapterContent, index) => {
        if (chapterContent.trim() === '') return;

        let chapterTitle = `chapter_${index + 1}`;
        const firstLine = chapterContent.split('\n')[0];
        const match = firstLine.match(/^(.*)/);
        if (match && match[1].trim() !== '') {
            chapterTitle = match[1].trim().replace(/\*/g, '').replace(/ /g, '_').replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '');
        }

        // Prepend '## ' back to the chapter content for all but the first chapter
        const contentToWrite = index === 0 ? chapterContent : '## ' + chapterContent;

        const outputFileName = path.join(outputDir, `${chapterTitle}.md`);

        fs.writeFile(outputFileName, contentToWrite, 'utf8', (err) => {
            if (err) {
                console.error(`Error writing chapter ${chapterTitle}:`, err);
            } else {
                console.log(`Successfully wrote ${outputFileName}`);
                chapterCount++;
            }
        });
    });
    console.log(`Total chapters processed: ${chapterCount}`);
});