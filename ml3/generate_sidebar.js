const path = require('path');

const fs = require('fs');
const chapterDir = '/mnt/g/deeplearn/ml3/chapter';

const chapterFiles = fs.readdirSync(chapterDir)
    .filter(file => file.endsWith('.md'))
    .map(file => path.join(chapterDir, file));

const sidebarPath = '/mnt/g/deeplearn/ml3/_sidebar.md';

function slugify(text) {
    return text.toLowerCase()
        .replace(/\s+/g, '-')           // Replace spaces with -
        .replace(/[^\w\-]+/g, '')       // Remove all non-word chars
        .replace(/\-\-+/g, '-')         // Replace multiple - with single -
        .replace(/^-+/, '')             // Trim - from start of text
        .replace(/-+$/, '');            // Trim - from end of text
}

function generateSidebarEntry(level, title, filePath) {
    const indent = '    '.repeat(level - 1);
    const fileName = path.basename(filePath);
    const relativePath = `chapter/__${path.basename(filePath, '.md').toUpperCase()}__.md`;
    const anchor = slugify(title);
    return `${indent}* [${title}](${relativePath}#${anchor})`;
}

let sidebarContent = '';

chapterFiles.forEach(filePath => {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');

    let chapterTitle = '';
    let chapterNumber = '';

    lines.forEach(line => {
        // Match H1 for main chapter title (e.g., # Chapter 1)
        const h1Match = line.match(/^#\s*(.*)/);
        if (h1Match) {
            chapterTitle = h1Match[1].trim();
            // Try to extract chapter number if present
            const chapterNumMatch = chapterTitle.match(/CHAPTER\s*(\d+)\s*(.*)/i);
            if (chapterNumMatch) {
                chapterNumber = chapterNumMatch[1];
                chapterTitle = chapterNumMatch[2].trim();
            }
            sidebarContent += `* [${chapterNumber ? chapterNumber + '. ' : ''}${chapterTitle}](chapter/__${path.basename(filePath, '.md').toUpperCase()}__.md#${slugify(chapterTitle)})
`;
            return;
        }

        // Match H2 for main sections (e.g., ## Section Title)
        const h2Match = line.match(/^##\s*(.*)/);
        if (h2Match) {
            const title = h2Match[1].trim();
            sidebarContent += generateSidebarEntry(1, title, filePath) + '\n';
            return;
        }

        // Match H3 for subsections (e.g., ### Subsection Title)
        const h3Match = line.match(/^###\s*(.*)/);
        if (h3Match) {
            const title = h3Match[1].trim();
            sidebarContent += generateSidebarEntry(2, title, filePath) + '\n';
            return;
        }

        // Match H4 for sub-subsections (e.g., #### Sub-subsection Title)
        const h4Match = line.match(/^####\s*(.*)/);
        if (h4Match) {
            const title = h4Match[1].trim();
            sidebarContent += generateSidebarEntry(3, title, filePath) + '\n';
            return;
        }
    });
});

fs.writeFileSync(sidebarPath, sidebarContent, 'utf8');
console.log(`Sidebar generated at ${sidebarPath}`);