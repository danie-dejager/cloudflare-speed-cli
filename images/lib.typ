#let foreground = rgb("#c7c055").lighten(20%)
#let background = foreground.desaturate(70%).darken(80%)

#let logo = {
  set text(
    size: 100pt,
    font: "Libertinus Sans",
    fill: foreground,
  )
  set image(height: 90pt)

  stack(
    dir: ltr,
    spacing: .3em,
    image(
      bytes(
        read("speedrabbit.svg").replace(
          regex("fill:#\d{6}"),
          "fill:" + foreground.to-hex(),
        ),
      ),
    ),
    [cloudflare-speed-cli],
  )
}
