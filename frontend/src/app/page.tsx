"use client";

import { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FileUploader } from "@/components/file-uploader";
import Image from "next/image";

export default function Home() {
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [file, setFile] = useState<File | null>(null);
  const [gridDimensions, setGridDimensions] = useState({ rows: 3, cols: 6 });
  const [backgroundImages, setBackgroundImages] = useState<string[]>([]);
  const [isLoadingImages, setIsLoadingImages] = useState(true);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    if (!file) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      console.log(result);
      setPrediction(result.prediction);
    } catch (error) {
      console.error("Error making prediction:", error);
    } finally {
      setIsLoading(false);
    }
  }

  // Define handleResize as memoized function to prevent recreating it on each render
  const handleResize = useCallback(() => {
    const width = window.innerWidth;

    let newRows, newCols;

    if (width < 640) {
      // Mobile
      newRows = 3;
      newCols = 4;
    } else if (width < 1024) {
      // Tablet
      newRows = 4;
      newCols = 6;
    } else {
      // Desktop
      newRows = 6;
      newCols = 8;
    }

    // Only update state if dimensions actually change
    if (newRows !== gridDimensions.rows || newCols !== gridDimensions.cols) {
      setGridDimensions({ rows: newRows, cols: newCols });
    }
  }, [gridDimensions.rows, gridDimensions.cols]);

  // Set up resize listener only once
  useEffect(() => {
    // Set initial size
    handleResize();

    // Add resize listener
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [handleResize]);

  // Fetch random images only when grid dimensions actually change
  useEffect(() => {
    async function fetchRandomImages() {
      setIsLoadingImages(true);
      try {
        const response = await fetch(
          `/api/random-images?rows=${gridDimensions.rows}&cols=${gridDimensions.cols}`
        );

        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();
        setBackgroundImages(data.images);
      } catch (error) {
        console.error("Error fetching random images:", error);
        // Fallback: generate empty array
        setBackgroundImages(
          Array(gridDimensions.rows * gridDimensions.cols).fill("")
        );
      } finally {
        setIsLoadingImages(false);
      }
    }

    fetchRandomImages();
  }, [gridDimensions.rows, gridDimensions.cols]);

  // Generate grid cells with background images or color fallbacks
  const gridCells = Array.from({
    length: gridDimensions.rows * gridDimensions.cols,
  }).map((_, index) => {
    // Create a variation of colors for the grid (fallback)
    const hue = (index * 15) % 360;
    const hasImage = backgroundImages[index] && !isLoadingImages;

    return (
      <div
        key={index}
        className="relative w-full h-full overflow-hidden"
        style={{
          backgroundColor: `hsl(${hue}, 70%, 85%)`,
          transition: "all 0.5s ease-in-out",
        }}
      >
        {hasImage && (
          <Image
            src={backgroundImages[index]}
            width={100}
            height={100}
            alt={`Skibidi image ${index}`}
            className="absolute inset-0 w-full h-full object-cover opacity-75 hover:opacity-100 transition-opacity"
          />
        )}
      </div>
    );
  });

  return (
    <div className="relative min-h-screen w-full overflow-hidden">
      {/* Grid Background */}
      <div
        className="absolute inset-0 grid z-0"
        style={{
          gridTemplateColumns: `repeat(${gridDimensions.cols}, 1fr)`,
          gridTemplateRows: `repeat(${gridDimensions.rows}, 1fr)`,
        }}
      >
        {gridCells}
      </div>

      {/* Main Content */}
      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center p-24">
        <Card className="w-[350px] shadow-xl backdrop-blur-sm bg-white/90">
          <CardHeader>
            <CardTitle>Video Skibidi Scorer</CardTitle>
            <CardDescription>
              Upload a video to get a Skibidi score
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit}>
              <div className="grid w-full items-center gap-4">
                <FileUploader
                  maxFileCount={1}
                  maxSize={4 * 1024 * 1024}
                  onValueChange={(files) => setFile(files[0])}
                  disabled={isLoading}
                  accept={{
                    "video/*": [],
                  }}
                />
                <Button
                  type="submit"
                  className="w-full"
                  disabled={isLoading || !file}
                  variant="outline"
                >
                  {isLoading ? (
                    <span className="flex items-center gap-2">
                      <svg
                        className="animate-spin h-4 w-4"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        ></circle>
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        ></path>
                      </svg>
                      Processing...
                    </span>
                  ) : (
                    "Predict"
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
          {prediction && (
            <CardFooter className="flex flex-col items-start">
              <div className="font-medium">Result:</div>
              <div className="mt-2 p-3 bg-slate-100 w-full rounded-md">
                {prediction}
              </div>
            </CardFooter>
          )}
        </Card>
      </div>
    </div>
  );
}
