'use client'

import { cn } from "../lib/utils"
import { Slider } from "./ui/slider"
import * as React from "react"
import { basePath } from "../app/settings"

type SliderProps = React.ComponentProps<typeof Slider>

export function InteractiveRelightViewer({ className, ...props }: SliderProps) {
  const [img_idx, setImgIndex] = React.useState(10);
  const [exprIdx, setExprIndex] = React.useState(0);

  const exprIds = [750, 143, 681, 380, 710, 1638];

  return (
    <div className="grid grid-cols-3 gap-10 items-center w-full">
      <div className="col-span-1">
        <Slider
          defaultValue={[10]}
          max={64}
          step={1}
          className={cn("w-full", className)}
          onValueChange={(value: any) => {
            setImgIndex(value[0]);
          }}
        />
        <div className="mt-3 mb-10 text-center select-none">Env Map Rotation</div>

        <Slider
          defaultValue={[0]}
          max={5}
          step={1}
          // className={cn("w-full", className)}
          onValueChange={(value: any) => {
            setExprIndex(value[0]);
          }}
        />
        <div className="mt-3 text-center select-none">Expression</div>
      </div>

      <div className="col-span-2">
        <img src={`${basePath}/env_map/expr_${exprIds[exprIdx]}_light_${img_idx}.webp`} className="rounded-3xl aspect-square"></img>
      </div>
    </div>
  )
}