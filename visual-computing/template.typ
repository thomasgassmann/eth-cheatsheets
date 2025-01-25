// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", authors: (), date: none, body) = {
  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1", number-align: center, flipped: true, margin: 3em)
  set text(font: "Libertinus Serif", lang: "en", size: 10pt)

  // Set paragraph spacing.
  show par: set block(above: 0.75em, below: 0.75em)

  set heading(numbering: "1.1")
  set par(leading: 0.58em)
  
  show: columns.with(3, gutter: 1em)

  // Title row.
  align(center)[
    #block(text(weight: 700, 1.75em, title))
    #v(0.8em, weak: true)
    #date
  ]

  // Author information.
  pad(
    top: 0.3em,
    bottom: 0.3em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center)[
        *#author.name* \
        #author.email
      ])
    ),
  )
  align(center, text(size: 10pt)[
    This document is licensed under CC BY-SA 4.0. It may be distributed or modified, as long as the author and the license remain intact.
    
    Based on work by #link("https://github.com/XYQuadrat/eth-cheatsheets/tree/main/viscomp")[*jsteinmann*] and #link("https://typst.app/project/ruyA4kPNzRyNmyu3MZqxny")[*jhoffmann*]
  ])

  // Main body.
  set par(justify: true)
  //set text(size: 0.85em)

  body
}

#let colorbox(title: none, inline: true, breakable: true, color: blue, content) = {
  let colorOutset = 4pt
  let titleContent = if title != none {
    box(
      fill: silver,
      outset: (left: colorOutset - 1pt, rest: colorOutset),
      width: if inline { auto } else { 100% },
      radius: if inline { (bottom-right: 8pt) } else { 0pt },
      [*#title*]) + if inline { h(6pt) }
  }

  block(
    stroke: (left: 2pt + color),
    outset: colorOutset, 
    fill: silver.lighten(60%), 
    breakable: breakable,
    width: 100%,
    
    titleContent + content)
}

#let fitWidth(content) = {
  layout((size) => {
    style((styles) => {
      let measures = measure(content, styles)
      let scaleFactor = if measures.width > size.width { 100% * (size.width / measures.width) } else { 100% }

      // Scale does not yet affect layout. place it - hidden box to adjust layout
      let scaled = scale(x: scaleFactor, y: scaleFactor, content)
      place(scaled)
      hide(box(height: measures.height * scaleFactor))
    })
  })
}

#let slashCircle = symbol("\u{2298}")


/*
// Apply inline only if it minimizes height.
#let colorbox(title: none, inline: none, breakable: true, color: blue, content) = {
  style((styles) => {
      let meas1 = measure(colorbox_s(title: title, inline: true, breakable: breakable, color: color, content), styles)
      let meas2 = measure(colorbox_s(title: title, inline: false, breakable: breakable, color: color, content), styles)
      let inline = meas1.height < meas2.height
    colorbox_s(title: title, inline: inline, breakable: breakable, color: color, content)
  })
}
*/