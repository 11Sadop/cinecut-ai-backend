import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.on('console', lambda msg: print('BROWSER CONSOLE:', msg.type, msg.text))
        page.on('pageerror', lambda err: print('BROWSER UNCAUGHT ERROR:', err))

        print('Navigating to https://cinecut-ai-studio.vercel.app ...')
        await page.goto('https://cinecut-ai-studio.vercel.app', wait_until='networkidle')

        print('Clicking on Card 1 (card-purple)...')
        await page.click('.card-purple')
        await page.wait_for_timeout(1000)

        modal_visible = await page.is_visible('#tool-action-modal')
        print('Modal visible after clicking Card 1:', modal_visible)

        modal_display = await page.evaluate("() => { const m = document.getElementById('tool-action-modal'); return m ? getComputedStyle(m).display : 'null'; }")
        print('Modal computed style display:', modal_display)

        await browser.close()

if __name__ == '__main__':
    asyncio.run(run())
