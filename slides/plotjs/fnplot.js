function linspace(start, stop, n) {
    let step = (stop - start) / n;
    return [...Array(n).keys()].map(i => {
        return start + i * step
    })
}

function fnplot(xs, f, id='myplot') {
    const data = xs.map(x => {
        return {x: x, y: f(x)}
    })

    const plot = Plot.plot({
        marks: [
            Plot.ruleX([0]),
            Plot.ruleY([0]),
            Plot.lineY(data, {x: "x", y: "y"})
        ]
    })

    const div = document.querySelector('#' + id);
    div.append(plot);
}