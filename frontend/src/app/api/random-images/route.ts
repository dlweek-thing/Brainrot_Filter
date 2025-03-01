import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

// Function to recursively get all image files from the public/images directory
function getAllImages(directory: string): string[] {
  const imageExtensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"];
  let results: string[] = [];

  try {
    // Convert from public path to filesystem path
    const fullPath = path.join(process.cwd(), "public", directory);

    // Check if directory exists
    if (!fs.existsSync(fullPath)) {
      return results;
    }

    const items = fs.readdirSync(fullPath);

    for (const item of items) {
      const itemPath = path.join(fullPath, item);
      const relativePath = path.join(directory, item);

      // Check if it's a directory
      if (fs.statSync(itemPath).isDirectory()) {
        // Recursively get images from subdirectories
        results = results.concat(getAllImages(relativePath));
      } else {
        // Check if it's an image file
        const ext = path.extname(item).toLowerCase();
        if (imageExtensions.includes(ext)) {
          // Save path relative to /public
          results.push("/" + relativePath);
        }
      }
    }
  } catch (error) {
    console.error("Error reading image directory:", error);
  }

  return results;
}

// Function to get random items from an array
function getRandomItems(array: string[], count: number): string[] {
  // Clone the array to avoid modifying the original
  const shuffled = [...array];

  // Fisher-Yates shuffle algorithm
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  // Return the requested number of items
  return shuffled.slice(0, count);
}

export async function GET(request: NextRequest) {
  // Get query parameters
  const searchParams = request.nextUrl.searchParams;
  const rows = parseInt(searchParams.get("rows") || "3");
  const cols = parseInt(searchParams.get("cols") || "6");

  const count = rows * cols;

  // Get all images from the public/images directory
  const allImages = getAllImages("images");

  // Get random subset based on requested grid size
  const randomImages = getRandomItems(allImages, count);

  return NextResponse.json({ images: randomImages });
}
