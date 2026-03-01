#!/usr/bin/env node
/**
 * Update 2025 Aggregate CSV Files
 * 
 * Run this script whenever you update master-data.csv to regenerate
 * the 2025 aggregate files for Need States, FEI Platforms, and Micro Categories.
 * 
 * Usage: node update-2025-aggregates.js
 * 
 * This calculates:
 * - Weighted Average CAGR = Σ(CAGR × Engagement) / Σ(Engagement)
 * - Sum of Engagement = Σ(Engagement)
 */

const fs = require('fs');
const path = require('path');

const MASTER_FILE = 'master-data.csv';

// Normalize FEI platform names to handle inconsistent capitalization
function normalizeFeiPlatform(name) {
    if (!name) return '';
    const normalized = name.trim();
    const feiMap = {
        'enchanting the senses': 'Enchanting The Senses',
        'hydration for vitality': 'Hydration For Vitality',
        'enhancing wellspans': 'Enhancing Wellspans',
        'targeted health support': 'Targeted Health Support',
        'mood mastery': 'Mood Mastery',
        'steady mental fuel': 'Steady Mental Fuel'
    };
    return feiMap[normalized.toLowerCase()] || normalized;
}

// Parse number from string (handles %, commas, N/A)
function parseNumber(val) {
    if (!val || val === 'N/A' || val === '') return null;
    const cleaned = String(val).replace(/[,%]/g, '').trim();
    const num = parseFloat(cleaned);
    return isNaN(num) ? null : num;
}

// Calculate aggregates with weighted average CAGR and sum engagement
function calculateAggregates(data, groupByField, normalize = null) {
    const groups = {};
    
    data.forEach(d => {
        let key = d[groupByField];
        if (!key) return;
        
        // Apply normalization if provided
        if (normalize) {
            key = normalize(key);
        }
        
        if (!groups[key]) {
            groups[key] = { totalEng: 0, weightedCagrSum: 0, count: 0 };
        }
        
        groups[key].totalEng += d.eng2025;
        groups[key].weightedCagrSum += d.cagr2025 * d.eng2025;
        groups[key].count++;
    });
    
    return Object.entries(groups)
        .map(([name, group]) => ({
            name,
            cagr: group.weightedCagrSum / group.totalEng,
            engagement: group.totalEng,
            count: group.count
        }))
        .sort((a, b) => b.engagement - a.engagement); // Sort by engagement descending
}

function main() {
    console.log('='.repeat(50));
    console.log('Updating 2025 Aggregate CSV Files');
    console.log('='.repeat(50));
    
    // Check if master file exists
    if (!fs.existsSync(MASTER_FILE)) {
        console.error(`Error: ${MASTER_FILE} not found!`);
        process.exit(1);
    }
    
    // Read and parse master data
    const masterData = fs.readFileSync(MASTER_FILE, 'utf-8');
    const lines = masterData.trim().split('\n').slice(2); // Skip 2 header rows
    
    const categories = lines.map(line => {
        const cols = line.split('\t');
        return {
            name: cols[1]?.trim() || '',
            eng2025: parseNumber(cols[2]),
            cagr2025: parseNumber(cols[3]),
            needState: cols[7]?.trim() || '',
            microCategory: cols[13]?.trim() || '',
            feiPlatform: cols[14]?.trim() || ''
        };
    }).filter(d => d.name && d.eng2025 !== null && d.cagr2025 !== null);
    
    console.log(`\nLoaded ${categories.length} categories from ${MASTER_FILE}`);
    
    // Generate Need State 2025
    const needStateAggregates = calculateAggregates(categories, 'needState');
    let needStateCSV = 'Need-State\t2025 NS CAGR (Weighted Average)\t2025 NS Engagement (Sum) (Millions)\n';
    needStateAggregates.forEach(d => {
        needStateCSV += `${d.name}\t${d.cagr.toFixed(2)}%\t${d.engagement.toFixed(6)}\n`;
    });
    fs.writeFileSync('need-state-data-2025.csv', needStateCSV);
    console.log(`\n✓ Created need-state-data-2025.csv (${needStateAggregates.length} need states)`);
    needStateAggregates.forEach(d => {
        console.log(`  - ${d.name}: ${d.cagr.toFixed(1)}% CAGR, ${d.engagement.toFixed(1)}M engagement (${d.count} categories)`);
    });
    
    // Generate FEI 2025
    const feiAggregates = calculateAggregates(categories, 'feiPlatform', normalizeFeiPlatform);
    let feiCSV = 'Front-End Innovation Platform\t2025 FEI CAGR (Weighted Average)\t2025 FEI Engagement (Sum) (Millions)\n';
    feiAggregates.forEach(d => {
        feiCSV += `${d.name}\t${d.cagr.toFixed(3)}%\t${d.engagement.toFixed(6)}\n`;
    });
    fs.writeFileSync('fei-data-2025.csv', feiCSV);
    console.log(`\n✓ Created fei-data-2025.csv (${feiAggregates.length} FEI platforms)`);
    feiAggregates.forEach(d => {
        console.log(`  - ${d.name}: ${d.cagr.toFixed(1)}% CAGR, ${d.engagement.toFixed(1)}M engagement (${d.count} categories)`);
    });
    
    // Generate Micro Category 2025
    const mcAggregates = calculateAggregates(categories, 'microCategory')
        .filter(d => d.engagement > 0);
    let mcCSV = 'Micro-Category\t2025 MC CAGR (Weighted Average)\t2025 MC Engagement (Sum) (Millions)\n';
    mcAggregates.forEach(d => {
        mcCSV += `${d.name}\t${d.cagr.toFixed(2)}%\t${d.engagement.toFixed(6)}\n`;
    });
    fs.writeFileSync('mc-data-2025.csv', mcCSV);
    console.log(`\n✓ Created mc-data-2025.csv (${mcAggregates.length} micro categories with data)`);
    
    console.log('\n' + '='.repeat(50));
    console.log('Done! All 2025 aggregate files updated.');
    console.log('='.repeat(50));
}

main();
